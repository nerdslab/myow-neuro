
class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, trial=None):
        for t in self.transforms:
            x = t(x, trial)
        return x

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))
