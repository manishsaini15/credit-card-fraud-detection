import os
import pickle


def save_object(file_path, obj):
    """
    Save Python object as pickle file.
    """

    dir_path = os.path.dirname(
        file_path
    )

    os.makedirs(
        dir_path,
        exist_ok=True
    )

    with open(
        file_path,
        "wb"
    ) as file_obj:

        pickle.dump(
            obj,
            file_obj
        )


def load_object(file_path):
    """
    Load pickle object.
    """

    with open(
        file_path,
        "rb"
    ) as file_obj:

        return pickle.load(
            file_obj
        )