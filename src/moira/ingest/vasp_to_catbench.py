import shutil
import yaml
from collections import defaultdict
from pathlib import Path
from catbench.adsorption.data import vasp as catbench_vasp

from moira.config import get_config
from moira.ingest.catbench_coefficients import build_coeff_setting


def tag_to_ads(tag, tag_map):
    label = tag_map[tag]
    return label.replace("*", "")


def copy_selected_files(src_dir: Path, dst_dir: Path, filenames=("CONTCAR", "OSZICAR")):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src_file = src_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, dst_dir / name)


def read_E0_from_OSZICAR(file_path):
    """
    Read final energy (E0) from VASP OSZICAR file.

    Args:
        file_path: Path to OSZICAR file

    Returns:
        float: Final energy in eV
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1]

        energy = None
        for word in last_line.split():
            if word == "E0=":
                energy_index = last_line.split().index(word) + 1
                energy = last_line.split()[energy_index]
                energy = float(energy)
                break

        if energy is None:
            raise ValueError(f"Energy value not found in file: {file_path}")

        return energy

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while reading the file '{file_path}': {str(e)}"
        )


def main():
    cfg = get_config()
    source = cfg.ingest.source
    dest = cfg.ingest.catbench_folder
    if dest is None:
        raise ValueError("cfg.ingest.catbench_folder is not initialized")

    if (source / "mapping.yaml").is_file():
        mapping_root = source
    elif (source.parent / "mapping.yaml").is_file():
        mapping_root = source.parent
    else:
        raise FileNotFoundError(
            f"Could not find mapping.yaml in {source} or {source.parent}"
        )

    with (mapping_root / "mapping.yaml").open() as f:
        tag_map = yaml.safe_load(f)

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "gas").mkdir(parents=True, exist_ok=True)

    # --------------------
    # GAS CONVERSION
    # --------------------
    gas_src = mapping_root / "gas"
    gas_dst = dest / "gas"

    if gas_src.is_dir():
        for folder in gas_src.iterdir():
            if not folder.is_dir():
                continue

            tag = folder.name[:4]
            if tag not in tag_map:
                continue

            ads = tag_to_ads(tag, tag_map)
            new_name = f"{ads}gas"

            copy_selected_files(folder, gas_dst / new_name)

    # --------------------
    # SYSTEM CONVERSION
    # --------------------
    test_src = source
    if not test_src.is_dir():
        raise FileNotFoundError(f"System source folder does not exist: {test_src}")

    systems = defaultdict(list)

    for folder in test_src.iterdir():
        if not folder.is_dir():
            continue

        oszicar_path = folder / "OSZICAR"
        try:
            # Filter out directories with missing/corrupt OSZICAR before conversion.
            read_E0_from_OSZICAR(oszicar_path)
        except RuntimeError:
            continue

        parts = folder.name.split("-")
        if len(parts) < 3:
            continue

        tag_part = parts[-1]
        tag = tag_part[:4]
        system = "-".join(parts[:-1])
        systems[system].append((tag, folder.name))

    for system, entries in systems.items():
        system_dir = dest / system
        system_dir.mkdir(parents=True, exist_ok=True)

        config_counter = defaultdict(int)

        for tag, folder in entries:
            src_path = test_src / folder

            if tag == "0000":
                dst_path = system_dir / "slab"
                copy_selected_files(src_path, dst_path)
            else:
                ads = tag_to_ads(tag, tag_map)
                config_counter[ads] += 1
                config_index = config_counter[ads]

                dst_path = system_dir / ads / str(config_index)
                copy_selected_files(src_path, dst_path)

    coeff_setting = build_coeff_setting(
        tag_map=tag_map,
        elements=list(cfg.ingest.stoich.elements),
        basis_species=list(cfg.ingest.stoich.basis_species),
    )

    # CatBench's VASP preprocessor uses one parameter for both:
    # 1) input dataset directory traversal, and 2) output JSON filename stem.
    # Patch its path helpers so we can read from the real dataset path while
    # writing to a stable project-root output file.
    project_root = Path.cwd()
    output_dir = project_root / "data" / "raw_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cfg.ingest.dataset_name}_adsorption.json"

    catbench_vasp.get_raw_data_directory = lambda: str(output_dir)
    catbench_vasp.get_raw_data_path = lambda _benchmark_name: str(output_path)
    catbench_vasp.vasp_preprocessing(
        dataset_name=str(dest),
        coeff_setting=coeff_setting,
    )


if __name__ == "__main__":
    main()
