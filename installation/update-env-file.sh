# Records the current environment to a file.
# Packages installed from GitHub with pip install <git url> will not be recorded
# properly (i.e. the link can be omitted and just replaced with the version).
# In that case, you have to update this file to add commands that
# will fix the environment file.
# (you could also patch the file manually afterwards).
# Similarly the conda channels used to install packages may not be recorded properly
# if you used complex combinations of channels.
# In that case you also have to make the edits here or patch the file manually.

ENVIR_FILE="environment.yml"
conda env export --file "$ENVIR_FILE"

# Delete the path line.
sed -i.deleteme "$ d" "$ENVIR_FILE"
# Set the package to a local installation.
sed -i.deleteme "/template-project-name==/d" "$ENVIR_FILE"
# .deleteme is a trick to make sed work the same way on both Linux and OSX.
# https://stackoverflow.com/questions/5694228/sed-in-place-flag-that-works-both-on-mac-bsd-and-linux
rm "${ENVIR_FILE}.deleteme"
