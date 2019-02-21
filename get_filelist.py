# https://gist.github.com/hellpanderrr/4ecf37745273df374a56
# python get files in a folder

def get_filelist(path, extension=None,only_folders=False):
    '''Returns list of files in a given folder, without going further
    Parameters
    ---------
    extension: Collect only files with this extension
    
    only_folders: Collect only folders names
    '''
    filenames = []
    if not only_folders:
        for i in os.walk(path).next()[2]:
            if (extension):
                if os.path.splitext(i)[1] == extension :
                    filenames.append(os.path.join(path,i))
            else:            
                filenames.append(os.path.join(path,i))
    else:
        for i in os.walk(path).next()[1]:
            filenames.append(os.path.join(path,i))
    return filenames