import os
import logging
import platform


class FileUtils:
    @staticmethod
    def rearrange_files(folder: str):
        """rearrange_files(folder)
        Rename files in the given folder in sequential names.
        Format is {prefix}.{index}.{suffix}
        e.g.
            origin files:
              animals/
                - cat.jpg
                - dog.jpg
            rearranged:
              animals/
                - animals.1.jpg
                - animals.2.jpg
        """
        if not os.path.exists(folder):
            logging.error("Folder {} not exists".format(folder))
            exit(1)
        path_sep = "\\" if "Windows" in platform.platform() else "/"
        prefix = folder.split(path_sep)[-1]
        files = os.listdir(folder)
        cnt = 0
        for file in files:
            os.rename(os.path.join(folder, file), "{}.{}.{}".format(prefix, cnt, file.split(".")[-1]))
            cnt += 1
