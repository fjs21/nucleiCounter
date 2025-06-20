import os, platform, fnmatch

from settings import Settings


def fullPath(root, name='', binary=False):
    """Return full path - compatible with long Paths >255 characters and can process binary or str inputs."""
    if binary:
        if name == '':
            fullpath = '\\\\?\\'.encode('utf8', 'ignore') + os.path.join(os.getcwd().encode('utf8', 'ignore'), root)
        else:
            fullpath = '\\\\?\\'.encode('utf8', 'ignore') + os.path.join(os.getcwd().encode('utf8', 'ignore'), root,
                                                                         name)
    else:
        if platform.system() == 'Windows':
            # fullpath = '\\\\?\\' + os.path.join(os.getcwd(),root,name)
            fullpath = os.path.join(os.getcwd(), root, name)
        elif platform.system() == 'Darwin':
            fullpath = os.path.join(os.getcwd(), root, name)
        else:
            raise 'Platform not recognized'
    return fullpath


def fileModified(root, name, binary=False):
    """Returns file modified property."""
    fullpath = fullPath(root, name, binary)
    try:
        modified_time = os.stat(fullpath).st_mtime
        # print(f"File Found: {fullpath}, length: {len(fullpath)}")
    except FileNotFoundError:
        print(f"File not found: {fullpath}, length: {len(fullpath)}")
        raise FileNotFoundError()
    return modified_time


def find(pattern, path, binary=False, excluded_subfolder=''):
    """Find all files in path (and subfolders) matching pattern."""
    result = []
    for root, dirs, files in os.walk(path):
        if excluded_subfolder in dirs:
            dirs.remove(excluded_subfolder)
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append({'path': root, 'name': name, 'modified': fileModified(root, name, binary)})
    return result


def findNewestMarkerFile(path):
    """Find newest marker file in path folder."""
    markerFiles = find("*.xml", path)
    new_markerFile = []
    for markerFile in markerFiles:
        if not new_markerFile or markerFile['modified'] > new_markerFile['modified']:
            new_markerFile = markerFile
            return new_markerFile
    print(f"Could not find markerFile (.xml) in path '{path}'.")


def findOldestMarkerFile(path):
    """Find newest marker file in path folder."""
    markerFiles = find("*.xml", path)
    new_markerFile = []
    for markerFile in markerFiles:
        if not new_markerFile or markerFile['modified'] < new_markerFile['modified']:
            new_markerFile = markerFile
            return new_markerFile
    print(f"Could not find markerFile (.xml) in path '{path}'.")


def findMatchingMarkerFile(path, imgFile):
    """Find matching marker file when multiple composites and markers are in the same subdir."""
    import re

    imgFile_base, extension = os.path.splitext(imgFile)

    markerFiles = find("*.xml", path)
    for markerFile in markerFiles:
        # markerFile_index1 = re.split(r"_|\.", markerFile['name'])[2]
        # markerFile_index2 = re.split(r"_|\.", markerFile['name'])[4]
        # imgFile_index1 = re.split(r"_|\.", imgFile)[1]
        # imgFile_index2 = re.split(r"_|\.", imgFile)[3]
        # if markerFile_index1 == imgFile_index1 and markerFile_index2 == imgFile_index2:
        #     print(f"Found '{markerFile['name']}' for imgFile '{imgFile}' in path '{path}'.")
        #     return markerFile
        markerFile_pattern = r'CellCounter_(.*?)\.xml'
        markerFile_match = re.search(markerFile_pattern, markerFile['name'])
        if markerFile_match:
            markerFile_base = markerFile_match.group(1)
            if markerFile_base == imgFile_base:
                print(f"Found '{markerFile['name']}' for imgFile '{imgFile}' in path '{path}'.")
                return markerFile

    print(f"Could not find match for imgFile '{imgFile}' in path '{path}'.")
    return None


def chkName(name):
    """Given a binary path, replace any unicode characters."""
    name_ascii = name.decode('ascii', 'ignore')
    name_unicode = name.decode('utf8')

    if name_ascii != name_unicode:
        print(f"File/Folder name contains unicode: '{name_unicode}'")
        print(f"Renaming to '{name_ascii}'")
        os.rename(name, name_ascii.encode('utf8'))


def fix_unicode_filenames(folder):
    """Parse all files and folders and replace unicode characters."""
    settings = Settings()
    root = settings.experiments[folder]['root'].encode('utf8', 'ignore')
    # pattern = settings.folder_dict[folder]['pattern'].encode('utf8','ignore')

    print(f"** Checking folder ({folder}) for unicode files/folder **")
    for root, dirs, files in os.walk(root):
        for name in dirs:
            chkName(fullPath(root, name, binary=True))
        for name in files:
            chkName(fullPath(root, name, binary=True))

            # files = find(pattern, root, binary=True)
    # for file in files:
    #     path = file['path'] 
    #     imgFile = file['name']

    #     markerFiles = find("*.xml".encode('utf'),path,binary=True)
    #     for markerFile in markerFiles:
    #         chkName(fullPath(markerFile['path'],markerFile['name'],binary=True))
    print("** All done! This folder can now be analyzed. **")


def parseFileName(imgFile):
    """Extract well name from file name."""
    import re

    pattern = r'_[A-F][0-9]+_'
    matches = re.findall(pattern, imgFile)
    well = None
    if matches:
        # Assuming you want to extract the first match
        well = matches[0]
        # Remove the "_" characters from the extracted code
        well = well.replace("_", "")
    else:
        print(f"No well ID found for {imgFile}.")

    return well


def loadKerasModel(filename):
    """Load h5 model file."""
    from keras.models import load_model
    return load_model(filename)


def remove_folder_contents(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            remove_folder_contents(item_path)  # Recursively remove subfolders
            os.rmdir(item_path)


def create_empty_folder(folder_path):
    if os.path.exists(folder_path):
        remove_folder_contents(folder_path)
    else:
        os.makedirs(folder_path)


""" Function to automatically find the FIJI path. UNNECESSARY"""
# def find_fiji():
#     if platform.system() == 'Darwin':
#         # Name of the Fiji application bundle
#         fiji_app_name = "Fiji.app"
#
#         # Search for the Fiji.app folder in the /Applications directory
#         applications_path = "/Applications"
#         fiji_app_path = os.path.join(applications_path, fiji_app_name)
#         # Check if the Fiji.app folder exists
#         if os.path.exists(fiji_app_path):
#             print("Path to Fiji.app:", fiji_app_path)
#             return fiji_app_path
#         else:
#             print("Fiji.app not found in the default Applications folder.")
#             raise "Fiji not found"
#     elif platform.system() == 'Windows':
#         # Name of the Fiji application folder
#         fiji_folder_name = "Fiji.app"
#
#         # Search for the Fiji.app folder in the Program Files directory
#         program_files_path = os.environ["ProgramFiles"]
#         fiji_folder_path = os.path.join(program_files_path, fiji_folder_name)
#
#         # Check if the Fiji.app folder exists
#         if os.path.exists(fiji_folder_path):
#             print("Path to Fiji.app:", fiji_folder_path)
#             return fiji_folder_path
#         else:
#             print("Fiji.app not found in the Program Files directory.")
#             raise "Fiji not found in Program Files folder"
