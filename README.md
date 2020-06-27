<img src='./art/syfertext_logo_horizontal.png'>
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

![CI](https://github.com/OpenMined/SyferText/workflows/CI/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


## Installation

In order to install and start using SyferText, you first have to install `git-lfs` by following [this short guide](https://github.com/git-lfs/git-lfs/wiki/Installation). 

Then go ahead and install our experimental language model that we adapted form spaCy's `en_core_web_lg` model. This should take a few minutes since the model size is >800M.

```
$ pip install git+git://github.com/Nilanshrajput/syfertext_en_core_web_lg@master
```

If you had already installed `syfertext_en_core_web_lg` prior to installing `git-lfs` please do the following:

1. Uninstall `syfertext_en_core_web_lg`
2. Install `git-lfs`.
3. Reinstall `syfertext_en_core_web_lg`.

Now you can go ahead and install SyferText:

```
$ git clone https://github.com/OpenMined/SyferText.git
$ cd SyferText
$ python setup.py install
```

That's it, you are good to go!

## Getting Started

SyferText can be used to work with datasets residing on a local machine (or a local worker as we call it in [PySyft](https://github.com/OpenMined/PySyft)), as well as with private datasets on remote workers. Here is a list of tutorials that you can follow to get more familiar with SyferText:

<table>
<tbody>
<tr>
<td align = 'center'>Code Examples</td>
<td align = 'center'>Use Cases</td>
</tr>
<tr>
<td>1. <a href= "https://github.com/OpenMined/SyferText/blob/master/tutorials/Part%200%20-%20(Getting%20Started)%20Local%20Tokenization.ipynb">Tokenizing local strings</a></td>
<td>1. <a href= "https://github.com/OpenMined/SyferText/blob/master/tutorials/usecases/UC01%20-%20Sentiment%20Classifier%20-%20Private%20Datasets%20-%20(Secure%20Training).ipynb">Training a sentiment classifier on multiple private datasets</a></td>
</tr>
<tr>
<td>2. <a href= "https://bit.ly/37VEJ28">Tokenizing remote strings</a></td>
</tr>
<tr>
<td>3. <a href= "https://github.com/OpenMined/SyferText/blob/master/tutorials/Part%202%20-%20(Getting%20Started)%20Using%20SimpleTagger.ipynb">Using the SimpleTagger</a></td>
</tr>
</tbody>
</table>


More tutorials are coming soon. Stay tuned!

## Our Team

SyferText is created and maintained by the NLP team at OpenMined and by volunteer contributors from all around the world. Here are the current members of the core NLP team. The team is growing!

<br>
<table>
  <tr>
    <td align="center">
      <a href="https://twitter.com/alan_aboudib">
        <img src="https://avatars1.githubusercontent.com/u/11991643?s=240" width="170px;" alt="Alan Aboudib avatar">
        <br /><sub><b>Alan Aboudib</b></sub></a><br />
        <sub>Team Lead / Author</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/dzlab">
        <img src="https://avatars0.githubusercontent.com/u/1645304?s=400&v=4" width="170px;" alt="Bachir Chihani avatar">
        <br /><sub><b>Bachir Chihani</b></sub></a><br />
        <sub>OM NLP Team / Contributor</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/MarcioPorto">
        <img src="https://avatars1.githubusercontent.com/u/6521281?s=400&v=4" width="170px;" alt="Marcio Porto avatar">
        <br /><sub><b>Márcio Porto</b></sub></a><br />
        <sub>OM NLP team / Contributor</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Nilanshrajput">
        <img src="https://avatars0.githubusercontent.com/u/28673745?s=400&u=4573311779fc3cc924670e3e02108e35350c1f25&v=4"  width="170px;" alt="Nilansh Rajput avatar">
        <br /><sub><b>Nilansh Rajput</b></sub></a><br />
        <sub>OM NLP team / Contributor</sub>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/sachin-101">
        <img src="https://avatars1.githubusercontent.com/u/44168164?s=400&u=df1c9d775a3312cacd4b330f469773e23260eb28&v=4"  width="170px;" alt="Sachin Kumar avatar">
        <br /><sub><b>Sachin Kumar</b></sub></a><br />
        <sub>OM NLP team / Contributor</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/bicycleman15">
        <img src="https://avatars0.githubusercontent.com/u/47978882?s=400&u=521e48efe1a9a652f4449f64278b690aa27dfe03&v=4"  width="170px;" alt="Jatin Prakash avatar">
        <br /><sub><b>Jatin Prakash</b></sub></a><br />
        <sub>OM NLP team / Contributor</sub>
      </a>
    </td>
  </tr>
  
  
</table>
<br>

## SyferText News

To get news about feature and tutorial relseases:

Alan Aboudib: [@twitter](https://twitter.com/alan_aboudib)

 and join [#lib_syfertext](https://openmined.slack.com/archives/CUWDZMED9) channel on slack.


## Call for Partners

We, at the NLP team, are eager to learn about new real-world use-cases around which new features in SyferText could be built. 

If you think that SyferText, in its current state or by adding more features, could be useful to your research or company, please contact us as indicated below in the **Contact Us** section, and let us discuss how we can help.


## Contact Us

You can reach out to us by contacting Alan on one of the following channels:

 [LinkedIn](https://www.linkedin.com/in/ala-aboudib/) | [Slack](https://app.slack.com/client/T6963A864/DDKH3SXKL/user_profile/UDKH3SH8S) | [Twitter](https://twitter.com/alan_aboudib)
 
-------

For more updates on our activities at OpenMined and for getting exciting news and announcements about our different projects, you can join our rapidly growing community of 7000+ on [Slack](https://slack.openmined.org/). You can follow our [official twitter page](https://twitter.com/openminedorg), as well as OpenMined [founder's twitter page](https://twitter.com/iamtrask).


## Support
For support in using this library, please join the **#lib_syfertext** Slack channel. If you’d like to follow along with any code changes to the library, please join the **#code_syfertext** Slack channel. [Click here to join our Slack community!](https://slack.openmined.org)


## Contributors ✨
[CONTRIBUTORS.md](https://github.com/OpenMined/SyferText/blob/master/CONTRIBUTORS.md)

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
