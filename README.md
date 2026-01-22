# 📦 Deep Learning-based inverse Hessian Estimation in Multi-scale Full Waveform Inversion

    
<img width="2944" height="1404" alt="image" src="https://github.com/user-attachments/assets/f2e7b1e7-5251-46f7-9776-8607a4f324f0" />
  
  
## ℹ️ Overview

This repository presents a research study on deep learning–based inverse Hessian estimation for multiscale Full Waveform Inversion (FWI).
The goal of this work is to improve model update quality, convergence stability, and computational efficiency in FWI by replacing expensive or impractical second-order calculations with a data-driven approximation of the inverse Hessian.

FWI is a highly nonlinear and ill-posed inverse problem. While Newton-type methods offer fast convergence by leveraging second-order information, explicit Hessian construction and inversion are computationally prohibitive for large-scale seismic problems. As a result, most practical workflows rely on gradient-based or Gauss–Newton approximations, which often suffer from slow convergence, amplitude imbalance, and sensitivity to cycle skipping—especially at higher frequencies.

This work introduces a deep learning framework that learns an approximate inverse Hessian operator directly from multiscale FWI updates. The learned operator acts as a preconditioner that transforms raw gradients into more physically meaningful and better-scaled model updates, effectively mimicking the behavior of second-order optimization while maintaining computational feasibility.

## ♠︎ Methods

#### Forward Modeling with Absorbing Boundary

The forward wave propagation is governed by the damped acoustic wave equation:

$$
\frac{1}{v^2}\frac{\partial^2 u(x,t)}{\partial t^2} - \nabla^2 u(x,t) + \eta \frac{\partial u(x,t)}{\partial t} = f_s(x,t)
$$

#### Misfit Calculation

The commonly used least-squares misfit function in FWI is defined as:

$$
E(m) = \frac{1}{2}\sum_{i=1}^{N}\left\| d_{\mathrm{syn}}^{\}(m) - d_{\mathrm{obs}}^{\} \right\|_2^2
$$

#### Adjoint Modelling

Using the adjoint-state method, the gradient is computed by correlating the forward wavefield with the adjoint wavefield. The adjoint modelling formulation is:

$$
\frac{\partial E(m)}{\partial m} = -\frac{2}{v^{3}} \int \frac{\partial^{2}u(x,t)}{\partial t^{2}} \ u^{*}(T-t)\,dt
$$

#### Model Update

After computing the model perturbation $\delta m$, the model is updated at iteration $k$ as:

$$
m_{k+1} = m_k + \delta m
$$

#### Model Perturbation Using the Hessian

In a Newton-type optimization framework, the model perturbation is computed using the inverse Hessian as:

$$
\delta m = - \left(\frac{\partial^{2} E(m)}{\partial m^{2}}\right)^{-1} \frac{\partial E(m)}{\partial m}
$$

## 🗂️ Folder Structure

This repository is organized as follows:  

- 🗂️ **devito_ta/**: Primary python files including the creation of model and geometry, plotting, preprocessing, and simulations.  
- 🗂️ **deeplearning_ta/**: Deep learning architectures to test.  
- 🗂️ **fwi_multiscale/**: notebooks for multi-scale FWI.  
- 🗂️ **fwi_singlescale/**: notebooks for single-scale FWI.  
  
  
## ✍️ Authors

Dziand Dafi Ginandjar - Institut Teknologi Bandung  
Andri Hendriyana - Institut Teknologi Bandung  
Infall Syafalni - Institut Teknologi Bandung  
  
  
## 💭 Feedback and Contributing
  
Multiscale implementation on 1 architecture ([EarthDoc paper](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202576026)).

