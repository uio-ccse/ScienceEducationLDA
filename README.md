# ScienceEducationLDA

**Work in progress**

Public repository for the Science Education LDA Project

## Description
This is the public repository for the Science Education LDA research project, which is maintained by [Tor Ole Odden](https://www.mn.uio.no/fysikk/english/people/aca/Tor%20Ole%20Odden/) and [Alessandro Marin](https://www.mn.uio.no/fysikk/english/people/adm/almarin/index.html).

This project is based on the method published in *Physical Review Physics Education Research* <sup><a href="#paper1">1</a></sup>. Also refer to the [CCSE/PERC_TopicModel](https://github.com/uio-ccse/PERC_TopicModel) repository. 


## Jupyter Notebook
See the [Science Education LDA Notebook](blob/master/Science%20Education%20LDA%20Notebook.ipynb) notebook. This notebook contains an extract of the methods described in <sup><a href="#paper1">1</a></sup>. 



## Installation
To run the main notebook PERC_TopicModeling.ipynb install the required packages: 

`pip install -r requirements.txt --user` 

A file (scied_words_bigrams_V5.pkl) contains the corpus obtained after processing the papers should be downloaded separately. Its size is about 200MB and the link will be posted soon. 

The required packages include Gensim (unsupervised semantic modelling on text), NLTK (Natural Language Tool Kit), LDAVis (interactive topic model visualization), scikit-learn, along with standard data analysis libraries such as pandas, numpy, and matplotlib.



## Preliminary Results
Graph of average topic prevalence over time: [AvgPrev.html](AvgPrev.html)

Graph of cumulative topic prevalence over time: [CumuPrev.html](ScienceEducationLDA)



## Contact
Questions can be directed to [Tor Ole Odden](https://www.mn.uio.no/fysikk/english/people/aca/Tor%20Ole%20Odden/)



## Literature

<a id='paper1'>1</a>: Tor Ole B. Odden and Alessandro Marin, Marcos D. Caballero. Thematic Analysis of 18 Years of Physics Education Research Conference Proceedings using Natural Language Processing, *Physical Review Physics Education Research*, 2020. [Link](https://journals.aps.org/prper/abstract/10.1103/PhysRevPhysEducRes.16.010142)