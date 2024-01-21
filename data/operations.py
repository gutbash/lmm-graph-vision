import yaml
import uuid

def add_object(file_path: str = None, text: str = None, image_data: str = None, image_path: str = None, expected: str = None, structure: str = None, generation: int = None, variation: int = None, format: int = None):
    
    if file_path is None:
        raise Exception('No file path provided.')
    
    new_object = {
        'id': str(uuid.uuid4()),
        'text': text,
        'image_data': image_data,
        'image_path': image_path,
        'expected': expected,
        'structure': structure,
        'generation': generation,
        'variation': variation,
        'format': format,
    }
    
    try:
        # Read the existing data from the file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file) or []

        # Append the new object
        data.append(new_object)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)

        print("Object added successfully to the YAML file.")
    except Exception as e:
        print(f"An error occurred: {e}")