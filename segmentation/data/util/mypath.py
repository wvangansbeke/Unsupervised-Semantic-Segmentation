#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os


class Path(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/path/to/PASCAL_VOC/' # VOC will be automatically downloaded
        db_names = ['VOCSegmentation']

        if database == '':
            return db_root

        if database in db_names:
            return os.path.join(db_root, database)

        else:
            raise ValueError('Invalid database {}'.format(database))
