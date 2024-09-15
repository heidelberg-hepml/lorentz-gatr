import glob


def to_filelist(flist):
    # keyword-based: 'a:/path/to/a b:/path/to/b'
    file_dict = {}
    for f in flist:
        if ":" in f:
            name, fp = f.split(":")
        else:
            name, fp = "_", f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    filelist = sum(file_dict.values(), [])
    assert len(filelist) == len(set(filelist))
    return file_dict, filelist
