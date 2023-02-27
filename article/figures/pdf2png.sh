#!/bin/bash
for pdfile in *.pdf ; do
  convert -verbose -density 300  "${pdfile}" "${pdfile%.*}".png
done

