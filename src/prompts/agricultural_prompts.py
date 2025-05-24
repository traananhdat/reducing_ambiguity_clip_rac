# src/prompts/agricultural_prompts.py

def get_basic_class_prompts(class_names):
    """
    Generates basic prompts for each class name.
    Example: "a photo of a Fuji apple"

    Args:
        class_names (list of str): A list of class names.
                                   e.g., ["fuji_apple", "cavendish_banana"]

    Returns:
        dict: A dictionary where keys are class names and values are lists of prompts.
    """
    prompts_dict = {}
    for name in class_names:
        # Replace underscores with spaces for more natural language
        natural_name = name.replace('_', ' ')
        prompts_dict[name] = [
            f"a photo of a {natural_name}",
            f"an image of a {natural_name}",
            f"a picture of a {natural_name}",
            f"a close-up photo of a {natural_name}",
            f"a {natural_name}",
        ]
    return prompts_dict

def get_authenticity_prompts(product_name, is_authentic_class_name, is_counterfeit_class_name):
    """
    Generates prompts specifically for distinguishing authentic vs. counterfeit
    for a given product.

    Args:
        product_name (str): The natural name of the product (e.g., "Fuji apple").
        is_authentic_class_name (str): The class name representing the authentic product
                                       (e.g., "authentic_fuji_apple").
        is_counterfeit_class_name (str): The class name representing the counterfeit product
                                         (e.g., "counterfeit_fuji_apple").

    Returns:
        dict: A dictionary where keys are the authentic/counterfeit class names
              and values are lists of corresponding prompts.
    """
    prompts_dict = {
        is_authentic_class_name: [
            f"a photo of an authentic {product_name}",
            f"an image of a genuine {product_name}",
            f"a picture of a real {product_name}",
            f"an authentic {product_name}",
            f"a genuine {product_name}, not counterfeit",
            f"this is a real {product_name}",
        ],
        is_counterfeit_class_name: [
            f"a photo of a counterfeit {product_name}",
            f"an image of a fake {product_name}",
            f"a picture of an imitation {product_name}",
            f"a counterfeit {product_name}",
            f"a fake {product_name}, not authentic",
            f"this is a non-genuine {product_name}",
            f"an imitation of a {product_name}",
        ]
    }
    return prompts_dict

def get_quality_prompts(product_name, quality_levels_map):
    """
    Generates prompts for different quality levels of a product.

    Args:
        product_name (str): The natural name of the product (e.g., "tomato").
        quality_levels_map (dict): A dictionary where keys are class names
                                   (e.g., "tomato_grade_a") and values are
                                   descriptors of that quality (e.g., "high quality", "grade A").

    Returns:
        dict: A dictionary where keys are quality class names and values are lists of prompts.
    """
    prompts_dict = {}
    for class_name, quality_descriptor in quality_levels_map.items():
        prompts_dict[class_name] = [
            f"a photo of a {quality_descriptor} {product_name}",
            f"an image of a {product_name} that is {quality_descriptor}",
            f"a {product_name} of {quality_descriptor} quality",
            f"a {quality_descriptor} {product_name}",
        ]
    return prompts_dict


def generate_prompts_for_task(task_type, class_config):
    """
    Generates a dictionary of prompts based on the task type and class configuration.

    Args:
        task_type (str): Type of task, e.g., "basic_classification",
                         "authenticity_check", "quality_grading".
        class_config (dict or list): Configuration specific to the task.
            - For "basic_classification": A list of class names.
            - For "authenticity_check": A dict like
              {'product_natural_name': 'Fuji Apple',
               'authentic_class_name': 'fuji_apple_real',
               'counterfeit_class_name': 'fuji_apple_fake'}
            - For "quality_grading": A dict like
              {'product_natural_name': 'Tomato',
               'levels': {'tomato_grade_a': 'Grade A', 'tomato_grade_b': 'Grade B'}}

    Returns:
        dict: A dictionary where keys are class names and values are lists of prompt strings.
              Returns None if task_type is unknown.
    """
    if task_type == "basic_classification":
        if not isinstance(class_config, list):
            raise ValueError("class_config must be a list of class names for basic_classification.")
        return get_basic_class_prompts(class_config)

    elif task_type == "authenticity_check":
        if not isinstance(class_config, dict) or not all(k in class_config for k in ['product_natural_name', 'authentic_class_name', 'counterfeit_class_name']):
            raise ValueError("Invalid class_config for authenticity_check.")
        return get_authenticity_prompts(
            class_config['product_natural_name'],
            class_config['authentic_class_name'],
            class_config['counterfeit_class_name']
        )

    elif task_type == "quality_grading":
        if not isinstance(class_config, dict) or not all(k in class_config for k in ['product_natural_name', 'levels']):
            raise ValueError("Invalid class_config for quality_grading.")
        return get_quality_prompts(
            class_config['product_natural_name'],
            class_config['levels'] # levels is a dict: {'class_name_grade_a': 'Grade A description'}
        )
    else:
        print(f"Warning: Unknown task_type '{task_type}' for prompt generation.")
        return None

# --- Example Usage (can be run directly for testing) ---
if __name__ == '__main__':
    print("--- Testing Basic Class Prompts ---")
    basic_classes = ["fuji_apple", "cavendish_banana", "organic_carrot"]
    basic_prompts = generate_prompts_for_task("basic_classification", basic_classes)
    if basic_prompts:
        for cn, p_list in basic_prompts.items():
            print(f"Class: {cn}, Example Prompt: {p_list[0]}")

    print("\n--- Testing Authenticity Prompts ---")
    auth_config = {
        'product_natural_name': "Hass Avocado",
        'authentic_class_name': "hass_avocado_authentic",
        'counterfeit_class_name': "hass_avocado_counterfeit"
    }
    auth_prompts = generate_prompts_for_task("authenticity_check", auth_config)
    if auth_prompts:
        for cn, p_list in auth_prompts.items():
            print(f"Class: {cn}, Example Prompt: {p_list[0]}")

    print("\n--- Testing Quality Prompts ---")
    quality_config = {
        'product_natural_name': "Strawberry",
        'levels': {
            "strawberry_premium": "premium grade",
            "strawberry_standard": "standard grade",
            "strawberry_grade_c": "grade C, suitable for processing"
        }
    }
    quality_prompts = generate_prompts_for_task("quality_grading", quality_config)
    if quality_prompts:
        for cn, p_list in quality_prompts.items():
            print(f"Class: {cn}, Example Prompt: {p_list[0]}")

    print("\n--- Testing Unknown Task Type ---")
    unknown_prompts = generate_prompts_for_task("ripeness_check", basic_classes)
    assert unknown_prompts is None

    print("\nPrompt generation tests completed.")