---
title: "Three-dimensional Segmentation of the Scoliotic Spine from MRI using Unsupervised Volume-based MR-CT Synthesis"
authors:
- admin
- Catherine Laporte
- Farida Cheriet
# author_notes:
# - "Equal contribution"
# - "Equal contribution"
date: "2020-10-12T00:00:00Z"
# doi: "10.1121/10.0000891"

# Schedule page publish date (NOT publication's date).
# publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "In *Proceedings of the SPIE Medical Imaging Conference 2021* San Diego, CA."
publication_short: ""

abstract: Vertebral bone segmentation from magnetic resonance (MR) images is a challenging task. Due to the inherent nature of the modality to emphasize soft tissues of the body, common thresholding algorithms are ineffective in detecting bones in MR images. On the other hand, it is relatively easier to segment bones from CT images because of the high contrast between bones and the surrounding regions. For this reason, we perform a cross-modality synthesis between MR and CT domains for simple thresholding-based segmentation of the vertebral bones. However, this implicitly assumes the availability of paired MR-CT data, which is rare, especially in the case of scoliotic patients. In this paper, we present a completely unsupervised, fully three-dimensional (3D) cross-modality synthesis method for segmenting scoliotic spines. A 3D CycleGAN model is trained for an unpaired volume-to-volume translation across MR and CT domains. Then, the Otsu thresholding algorithm is applied to the synthesized CT volumes for easy segmentation of the vertebral bones. The resulting segmentation is used to reconstruct a 3D model of the spine. We validate our method on 28 scoliotic vertebrae in 3 patients by computing the point-to-surface mean distance between the landmark points for each vertebra obtained from pre-operative X-rays and the surface of the segmented vertebra. Our study results in a mean error of 3.41 $\pm$ 1.06 mm. Based on qualitative and quantitative results, we conclude that our method is able to obtain a good segmentation and 3D reconstruction of scoliotic spines, all after training from unpaired data in an unsupervised manner.

# Summary. An optional shortened abstract.
# summary: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis posuere tellus ac convallis placerat. Proin tincidunt magna sed ex sollicitudin condimentum.

tags:
- Source Themes
featured: false

# links:
# - name: ""
# url: 
url_pdf: https://arxiv.org/pdf/2011.14005.pdf
url_code: ''
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

