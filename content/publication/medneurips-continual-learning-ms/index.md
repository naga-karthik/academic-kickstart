---
title: "Segmentation of Multiple Sclerosis Lesions across Hospitals: Learn Continually or Train from Scratch?"
authors:
- admin
- Anne Kerbrat
- Pierre Labauge
- Tobias Granberg
- Jason Talbott
- Daniel S Reich
- Massimo Fillipi
- Rohit Bakshi
- Virginie Callot
- Sarath Chandar
- Julien Cohen-Adad
# author_notes:
# - "Equal contribution"
# - "Equal contribution"
date: "2022-11-12T00:00:00Z"
# doi: "10.1121/10.0000891"

# Schedule page publish date (NOT publication's date).
# publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "Medical Imaging Meets NeurIPS Workshop 2022, NeurIPS, New Orleans, LA, USA"
publication_short: ""

abstract: Segmentation of Multiple Sclerosis (MS) lesions is a challenging problem. Several deep-learning-based methods have been proposed in recent years. However, most methods tend to be static, that is, a single model trained on a large, specialized dataset, which does not generalize well. Instead, the model should learn across datasets arriving sequentially from different hospitals by building upon the characteristics of lesions in a continual manner. In this regard, we explore experience replay, a well-known continual learning method, in the context of MS lesion segmentation across multi-contrast data from 8 different hospitals. Our experiments show that replay is able to achieve positive backward transfer and reduce catastrophic forgetting compared to sequential fine-tuning. Furthermore, replay outperforms the multi-domain training, thereby emerging as a promising solution for the segmentation of MS lesions. The code is open-source and available at [this link](https://github.com/naga-karthik/continual-learning-ms).

# Summary. An optional shortened abstract.
# summary: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis posuere tellus ac convallis placerat. Proin tincidunt magna sed ex sollicitudin condimentum.

tags:
- Source Themes
featured: false

# links:
# - name: ""
# url: 
url_pdf: https://arxiv.org/pdf/2210.15091.pdf
url_code: https://github.com/naga-karthik/continual-learning-ms
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# image:
#  caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/jdD8gXaTZsc)'
#  focal_point: ""
#  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: example
---

