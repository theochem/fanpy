# Install git hooks for the project.

DIR=$(dirname $(readlink -f ${BASH_SOURCE:-0}))
cd ${DIR}
for HOOK in *.hook
do
    ln -s ${DIR}/${HOOK} ${DIR}/../.git/hooks/$(echo ${HOOK} | sed "s/.hook$//")
done
