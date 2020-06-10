#!/bin/bash
make html
cd _build/html
git add .
git commit -m 'rebuilt docs'
git push origin gh-pages
