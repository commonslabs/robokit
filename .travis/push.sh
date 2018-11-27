#!/bin/sh

git config --global user.email "travis@travis-ci.org"
git confit --global user.name "Travis CI"

git checkout -b pre-deploy

python3 py2md.py --sourcedir ./robokit/ --docfile docs.md -projectname "Robokit" --codelinks

git commit --message "Travis Build: $TRAVIS_BUILD_NUMBER"

git remote add origin https://${GH_TOKEN}@github.com/Thomascountz/robokit.git

git push --setup-upstream origin pre-deploy
