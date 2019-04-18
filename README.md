# Using Convolutional Neural Networks for Classification of Malware represented as Images

This code base is no longer maintained and exists as a historical artifact to supplement my
 [Master's thesis](https://upcommons.upc.edu/handle/2117/91319) and the paper 
 [Using Convolutional Neural Networks for Classification of Malware represented as Images](https://link.springer.com/article/10.1007/s11416-018-0323-0).


## Requirements

Code is written in Python 2.7 and requires Tensorflow 1.9.0.

## Citing 
If you find this work useful in your research, please consider citing:
```
@Article{Gibert2019,
    author="Gibert, Daniel
    and Mateu, Carles
    and Planes, Jordi
    and Vicens, Ramon",
    title="Using convolutional neural networks for classification of malware represented as images",
    journal="Journal of Computer Virology and Hacking Techniques",
    year="2019",
    month="Mar",
    day="01",
    volume="15",
    number="1",
    pages="15--28",
    abstract="The number of malicious files detected every year are counted by millions. One of the main reasons for these high volumes of different files is the fact that, in order to evade detection, malware authors add mutation. This means that malicious files belonging to the same family, with the same malicious behavior, are constantly modified or obfuscated using several techniques, in such a way that they look like different files. In order to be effective in analyzing and classifying such large amounts of files, we need to be able to categorize them into groups and identify their respective families on the basis of their behavior. In this paper, malicious software is visualized as gray scale images since its ability to capture minor changes while retaining the global structure helps to detect variations. Motivated by the visual similarity between malware samples of the same family, we propose a file agnostic deep learning approach for malware categorization to efficiently group malicious software into families based on a set of discriminant patterns extracted from their visualization as images. The suitability of our approach is evaluated against two benchmarks: the MalImg dataset and the Microsoft Malware Classification Challenge dataset. Experimental comparison demonstrates its superior performance with respect to state-of-the-art techniques.",
    issn="2263-8733",
    doi="10.1007/s11416-018-0323-0",
    url="https://doi.org/10.1007/s11416-018-0323-0"
}
```