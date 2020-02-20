#!/bin/bash
PROCESS_TO_KILL="RNAnet.py"
PROCESS_LIST=`ps ax | grep -Ei ${PROCESS_TO_KILL} | grep -Eiv '(grep|vi RNAnet.py)' | awk ' { print $1;}'`
KILLED=
for KILLPID in $PROCESS_LIST; do
  if [ ! -z $KILLPID ];then
    kill -9 $KILLPID
    echo "Killed PID ${KILLPID}"
    KILLED=yes
  fi
done

if [ -z $KILLED ];then
    echo "Didn't kill anything"
fi
