from collections import namedtuple

Label = namedtuple('Label', ['name', 'id', 'train_id', 'color', ])


def label_creator(valid_classes=None, merge_to_one=False, binary_color=(255, 0, 0), ignore_index=255):
    """
    Label creator closure which parses input valid classes and binary segmentation term.
    :param valid_classes: indices of classes that will be used in training. If is not set indices not equal to
                          ignore_index are treated as valid
    :param merge_to_one: bool, if the valid classes should be treated as one (i.e. binary segmentation)
    :param ignore_index: int, set to the train_id of not valid classes
    :param binary_color: mask color will be used in case of binary segmentation
    :returns Label creator function, which creates label of type Label with following properties:
             1. If the label is in valid_classes list it obtains train_id in order of definition, otherwise it is
                initialized with the ignore_index.
             2. If binary mode is set to True, then all the valid classes are treated as one, and obtain train_id = 1
                and color = binary_color

    """
    counter = 0
    def add_label(name, id, train_id, color):
        if valid_classes:
            if id in valid_classes:
                nonlocal counter
                train_id = counter
                counter += 1
            else:
                train_id = ignore_index
        if merge_to_one and train_id != ignore_index:
            train_id = 1
            color = binary_color
        return Label(name, id, train_id, color)
    return add_label