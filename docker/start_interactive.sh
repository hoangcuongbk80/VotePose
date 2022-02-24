#!/bin/bash

if [ -z $1 ];
        then
        echo "You must provide a docker volume to mount as argument";
        exit;
fi

USER=`whoami`
echo $USER

nvidia-docker run --name my_votepose -it -v $1:/mnt/data -v /home/dhg/Votepose/:/Votepose --privileged --cap-add=SYS_ADMIN --label user=$USER --ipc=host hoangcuongbk80/votepose /bin/bash
