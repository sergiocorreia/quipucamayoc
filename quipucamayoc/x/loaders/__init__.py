from .doc3dwc_loader import doc3dwcLoader
from .doc3dbmnoimgc_loader import doc3dbmnoimgcLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'doc3dwc':doc3dwcLoader,
        'doc3dbmnic':doc3dbmnoimgcLoader,
    }[name]
