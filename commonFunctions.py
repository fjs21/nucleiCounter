import os, fnmatch

from settings import Settings

def fullPath(root, name='', binary=False):
    """Return full path - compatible with long Paths >255 characters and can process binary or str inputs."""
    if binary:
        if name =='':
            fullpath = '\\\\?\\'.encode('utf8','ignore') + os.path.join(os.getcwd().encode('utf8','ignore'),root)
        else:
            fullpath = '\\\\?\\'.encode('utf8','ignore') + os.path.join(os.getcwd().encode('utf8','ignore'),root,name)
    else:
        fullpath = '\\\\?\\' + os.path.join(os.getcwd(),root,name)
    return fullpath

def fileModified(root, name, binary=False):
    """Returns file modified property."""
    fullpath = fullPath(root,name,binary)
    try:
        modified_time = os.stat(fullpath).st_mtime
        #print(f"File Found: {fullpath}, length: {len(fullpath)}")
    except FileNotFoundError:
        print(f"File not found: {fullpath}, length: {len(fullpath)}")
        raise FileNotFoundError()
    return modified_time

def find(pattern, path, binary=False):
    """Find all files in path (and subfolders) matching pattern."""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append({'path':root,'name':name,'modified':fileModified(root, name, binary)})
    return result

def findNewestMarkerFile(path):
    """Find newest marker file in path folder."""
    markerFiles = find("*.xml",path)
    new_markerFile = []
    for markerFile in markerFiles:
        if not new_markerFile or markerFile['modified'] > new_markerFile['modified']:
            new_markerFile = markerFile
            return new_markerFile
    print(f"Could not find markerFile (.xml) in path '{path}'.")

def findOldestMarkerFile(path):
    """Find newest marker file in path folder."""
    markerFiles = find("*.xml",path)
    new_markerFile = []
    for markerFile in markerFiles:
        if not new_markerFile or markerFile['modified'] < new_markerFile['modified']:
            new_markerFile = markerFile
            return new_markerFile
    print(f"Could not find markerFile (.xml) in path '{path}'.")

def findMatchingMarkerFile(path, imgFile):
    """Find matching marker file when multiple composites and markers are in the same subdir."""
    import re
    markerFiles = find("*.xml",path)
    for markerFile in markerFiles:
        markerFile_index1 = re.split(r"_|\.", markerFile['name'])[2]
        markerFile_index2 = re.split(r"_|\.", markerFile['name'])[4]
        imgFile_index1 = re.split(r"_|\.", imgFile)[1]
        imgFile_index2 = re.split(r"_|\.", imgFile)[3]
        if markerFile_index1 == imgFile_index1 and markerFile_index2 == imgFile_index2:
            return markerFile
    print(f"Could not find match for imgFile '{imgFile}' in path '{path}'.")
    return []

def chkName(name):
    """Given a binary path, replace any unicode characters."""
    name_ascii = name.decode('ascii','ignore')
    name_unicode = name.decode('utf8')

    if name_ascii != name_unicode:
        print(f"File/Folder name contains unicode: '{name_unicode}'")
        print(f"Renaming to '{name_ascii}'")
        os.rename(name, name_ascii.encode('utf8')) 

def fix_unicode_filenames(folder):
    """Parse all files and folders and replace unicode characters."""
    settings = Settings()
    root = settings.folder_dicts[folder]['root'].encode('utf8','ignore')
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

def loadKerasModel(filename):
    """Load h5 model file."""
    from keras.models import load_model
    return load_model(filename)