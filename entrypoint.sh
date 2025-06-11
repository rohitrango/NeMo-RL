#!/bin/bash
# Entrypoint script to ensure a user with the Docker-specified UID exists
 
# Default username
USERNAME=rohitkumarj
 
# Use environment variables passed to the docker run command to set the uid and guid
# IMPORTANT NOTE: the uid and guid should be passed in as environment variables
# IMPORTANT NOTE: not via the --user flag in docker run, using --user will result
# IMPORTANT NOTE: in this script not being run as the root user, which is necessary for setting up sudo
USER_UID=${HOST_USER_UID:-1000}
USER_GID=${HOST_USER_GID:-1000}
 
# Create a new group with the USER_GID if it does not already exist
if ! getent group $USER_GID > /dev/null; then
    groupadd -g $USER_GID $USERNAME
fi
 
# Create a new user with the USER_UID and USER_GID if it does not already exist
if ! getent passwd $USER_UID > /dev/null; then
    useradd -l -u $USER_UID -g $USER_GID -m $USERNAME
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi
 
# Switch to the created user and execute the command
exec gosu $USERNAME "$@"
