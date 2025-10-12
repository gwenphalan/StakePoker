
import json
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, ValidationError

# Define the path to the regions file relative to the project root.
# This assumes 'src' is in the project root.
PROJECT_ROOT = Path(__file__).parent.parent.parent
REGIONS_FILE_PATH = PROJECT_ROOT / "data" / "regions.json"

class RegionModel(BaseModel):
    """
    Pydantic model for a single poker table region.
    Ensures that region data has the correct structure and types.
    """
    x: int
    y: int
    width: int
    height: int

def load_regions_typed() -> Dict[str, RegionModel]:
    """
    Loads poker region definitions from 'data/regions.json', validates them
    against the RegionModel, and returns them as typed objects.

    Returns:
        A dictionary mapping region names to validated RegionModel objects.

    Raises:
        FileNotFoundError: If 'data/regions.json' does not exist.
        ValidationError: If the region data is malformed.
    """
    if not REGIONS_FILE_PATH.exists():
        raise FileNotFoundError(f"Regions file not found at: {REGIONS_FILE_PATH}")

    with open(REGIONS_FILE_PATH, 'r') as f:
        raw_regions = json.load(f)

    try:
        validated_regions = {
            name: RegionModel(**data) for name, data in raw_regions.items()
        }
        print(f"Successfully loaded and validated {len(validated_regions)} regions from {REGIONS_FILE_PATH}")
        return validated_regions
    except ValidationError as e:
        print(f"Error validating regions configuration in {REGIONS_FILE_PATH}: {e}")
        raise

def save_regions_typed(regions: Dict[str, RegionModel]) -> None:
    """
    Saves a dictionary of RegionModel objects to 'data/regions.json'.

    Args:
        regions: A dictionary mapping region names to RegionModel objects.

    Raises:
        IOError: If the file cannot be written.
    """
    # Ensure the 'data' directory exists
    REGIONS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Convert Pydantic models back to dictionaries for JSON serialization
    regions_dict = {name: model.model_dump() for name, model in regions.items()}

    try:
        with open(REGIONS_FILE_PATH, 'w') as f:
            json.dump(regions_dict, f, indent=4)
        print(f"Regions saved successfully to {REGIONS_FILE_PATH}")
    except IOError as e:
        print(f"Error saving regions to {REGIONS_FILE_PATH}: {e}")
        raise
