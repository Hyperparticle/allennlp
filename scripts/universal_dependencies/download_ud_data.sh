#!/usr/bin/env bash

# Download UD 2.4
UD_2_4="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz?sequence=4&isAllowed=y"

DATA_DIR="${HOME}/.universal_dependencies"
ARCHIVE_PATH="${DATA_DIR}/ud_data.tgz"

if [[ ! -f ${ARCHIVE_PATH} ]]; then
    echo "Downloading UD data..."
    mkdir -p ${DATA_DIR}
    curl ${UD_2_4} -o ${ARCHIVE_PATH}
fi

echo "Extracting UD data..."
tar -xvzf ${ARCHIVE_PATH} -C ${DATA_DIR}
