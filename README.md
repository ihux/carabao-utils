# Carabao Utils

![Carabao Banner](doc/image/carabao.jpg)

## Carabao

_A carabao is a swamp-type domestic water buffalo. For a Philippine farmer a carabao is a source of draft animal power which is endlessly helpful for mastering the challenges of his daily life. A Carabao is a Python-type 'domestic' class object, and for a system-, process-, control-engineer or AI researcher a Carabao is a source of conceptual power which is endlessly helpful for mastering the rapid prototyping challenges of his daily life._

## Markdown Language for Documentation

This `README.md` file is written using easy-to-learn `Markdown` for text
formatting, hyperlink support, image embedding, and more. You can visit
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
for a `markdown` crashcourse.

On macOS we recommend [MacDown](https://macdown.uranusjr.com) as markdown editor,
but many editors (like Atom) do the job as well.

## Tools

For efficient work we need the following tools

* git
* github desktop
* python
* pip
* virtual environments
* jupyter lab
* colab

## How to Get Started

Before we go through the mentioned tools in detail we explain how you get started.

* Make sure that `git` and `Github Desktop` is installed on your computer, otherwise
install these tools (see below).

* We recommend the following file tree structure when working with `git
  repositories` (`repositories` or `repos` are the file packages being managed
  by git, which can be accessed on `github` by unique URLs):
```
└── git
    ├── repo-group1
    │   ├── repoA
    │   ├── repoB
    │   └── tmp
    ├── repo-group2
    │   ├── repoK
    │   ├── repoL
    │   ├── repoM
    │   └── tmp
    :
    ├── repo-groupN
    │   ├── repoX
    │   ├── repoY
    │   ├── repoZ
    │   └── tmp
```
This recommendation suggests a central root directory which you can give any name
(we name it `git`), under the root directory we arrange repository group
directories (`repo-group1`, `repo-group2`, ..., `repo-groupN`), and under each
repository group you install several repositories plus an optional `tmp` folder,
where you can store local copies for easy comparison.

In our case we choose the name `neural` for the repository group name, and would
do the following preparations in a command window:

```
   $ mkdir path-to/git       # the root directory of all our repos
   $ cd path-to/git          # cd into our repo root
   $ mkdir neural            # the group directory of all our `neural repos`
   $ cd neural               # cd into group directory
```

With current directory changed to our repo group `path-to/git/neural` we clone
our first repository from github by invoking:

```
   $ git clone https://github.com/ihux/carabao-utils.git
```


## Git

`Git` is a version control system that allows developers to track a project and actively contribute without interfering in each other’s work. `git` should be pre-installed on recent macOS systems. If not, get instructions from the official `git` web site
https://git-scm.com. With an installed `git` you can use the full function set of `git` from the command line.

## Github Desktop

While `git` is the tool to perform version control, we have `github` as a cloud site to backup the local data on a remote site. [Github Desktop](https://docs.github.com/en/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop) is an easy-to-use graphical tool around `git` which performs the `git` operations on mouse click.
