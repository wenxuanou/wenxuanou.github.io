---
title: 'Survey on ML model parameter scale and their deployments'
date: 2026-11-18
permalink: /posts/2026/11/survey-on-model-parameters/
tags:
  - machine learning
---

I did a small little survey on popular models about their parameters and deployment methods/requirements. I only consider the version of largest parameters of one's kind without quantization. Specs of close-sourced models are estimated. 

The first version of the table includes models available until 2026-11. The table will update later on.

{% include components/data-figure.html
   id="llm-params"
   src="/assets/data/llm-params.csv"
   type="timeline"
   x="total_params"
   y="active_params"
   series="deployment"
   label="model"
   title="ML Model Parameters"
   xlabel="parameters in log scale"
   height="470"
   caption="Activation parameter amount is labeled with saturated color for MoE models. Below is model applications. VRAM is measured in GB, parameters and active parameters are measured in billions(B)." 
%}





