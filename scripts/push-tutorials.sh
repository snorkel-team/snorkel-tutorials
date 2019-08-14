#!/bin/sh

setup_git() {
  git config --global user.email "54054737+snork2d2@users.noreply.github.com"
  git config --global user.name "snork2d2"
}

checkout_website() {
  git clone https://$GITHUB_TOKEN@github.com/snorkel-team/website.git > /dev/null 2>&1
}

build_tutorials() {
  rm -rf build website/_use_cases
  tox -e markdown
  mkdir -p website/_use_cases
  mv -f build/*.md website/_use_cases
}

push_tutorials() {
  cd website
  git add _use_cases
  git commit -m "[DEPLOY $TRAVIS_BUILD_NUMBER] Update tutorials"
  git push https://$GITHUB_TOKEN@github.com/snorkel-team/website.git master -f > /dev/null 2>&1
}

echo "Setting up git"
setup_git
echo "Checking out website"
checkout_website
echo "Building tutorial web pages"
build_tutorials
echo "Pushing tutorial web pages"
push_tutorials
