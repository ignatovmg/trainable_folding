#!/bin/bash

jskill $(jslist | grep Run | awk '{print $1}')
