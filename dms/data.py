def loadMatrix(path):
    """
    Reads a Blosum matrix from file. Changed slightly to read in larger blosum matrix
    File in a format like:
        https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62
    Input:
        path: str, path to a file.
    Returns:
        blosumDict: Dictionary, The blosum dict
    """

    with open(path, "r") as f:
        content = f.readlines()

    blosumDict = {}

    header = True
    for line in content:
        line = line.strip()

        # Skip comments starting with #
        if line.startswith(";"):
            continue

        linelist = line.split()

        # Extract labels only once
        if header:
            labelslist = linelist
            header = False

            # Check if all AA are covered
            #if not len(labelslist) == 25:
            #    print("Blosum matrix may not cover all amino-acids")
            continue

        if not len(linelist) == len(labelslist) + 1:
            print(len(linelist), len(labelslist))
            # Check if line has as may entries as labels
            raise EOFError("Blosum file is missing values.")

        # Add Line/Label combination to dict
        for index, lab in enumerate(labelslist, start=1):
            blosumDict[f"{linelist[0]}{lab}"] = float(linelist[index])

    # Check quadratic
    if not len(blosumDict) == len(labelslist) ** 2:
        print(len(blosumDict), len(labelslist))
        raise EOFError("Blosum file is not quadratic.", len(blosumDict), len(labelslist)**2)
    return blosumDict
