//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file frequencygrid.cpp
//  \brief implementation of functions in class Radiation
//======================================================================================

#include <sstream>  // msg
#include <stdio.h>  // fopen and fwrite

// Athena++ headers
#include "./radiation.hpp"


//--------------------------------------------------------------------------------------
// \!fn void FrequencyGrid()

// \brief function to create the frequency grid
// specific intensities are still defined as frequency integrated over each

void Radiation::FrequencyGrid()
{
  Real h_planck = 6.6260755e-27; // Planck constant
  Real k_b = 1.380649e-16;   // Boltzman constant

   // convert frequency to unit of kT_unit/h
  if(nu_min < 0.0)
    nu_min = -nu_min;
  else
    nu_min = nu_min * h_planck/(k_b * tunit);

  if(nu_max < 0.0)
    nu_max = -nu_max;
  else
    nu_max = nu_max * h_planck/(k_b * tunit);


  if(nu_max <= nu_min){
    std::stringstream msg;
    msg << "### FATAL ERROR in Radiation Class" << std::endl
        << "frequency_max needs to be larger than frequency_min!";
    throw std::runtime_error(msg.str().c_str());
  }

  if(log_fre_ == 1){
    if(fre_ratio > 1){

      if(nfreq > 2){
        nu_max = nu_min * pow(fre_ratio,nfreq-2);
      }else{
        nfreq = log10(nu_max/nu_min)/log10(fre_ratio)+2;
      }// calculate nfreq if not given

    }else{
      if(nfreq > 2){
        fre_ratio = log10(nu_max/nu_min)/(nfreq - 2);
        fre_ratio = pow(10.0,fre_ratio);
      }// end nfreq > 1
    }
  }// end log frequency spacing

  if(nfreq > 1){
    nu_grid.NewAthenaArray(nfreq);
    nu_grid(0) = 0.0;
    nu_grid(1) = nu_min;

    if(nfreq > 2){
      if(log_fre_ == 1){
        for(int n=2; n<nfreq; ++n)
          nu_grid(n) = nu_grid(n-1) * fre_ratio;
      }else{
        Real d_nu = (nu_max - nu_min)/(nfreq-2);

        for(int n=2; n<nfreq; ++n)
          nu_grid(n) = nu_grid(n-1) + d_nu;
      }
    }// end nfreq > 2

    nu_cen.NewAthenaArray(nfreq);

    for(int n=0; n<nfreq-1; ++n)
      nu_cen(n) = 0.5*(nu_grid(n)+nu_grid(n+1));

    delta_nu.NewAthenaArray(nfreq);

    for(int n=0; n<nfreq-1; ++n)
      delta_nu(n) = nu_grid(n+1)-nu_grid(n);

  }// end nfreq > 2
  emission_spec.NewAthenaArray(nfreq);

  // initialize with default emission spectrum assuming tgas=1
  if(nfreq == 1)
    emission_spec(0) = 1.0;
  else{
    for(int ifr=0; ifr<nfreq-1; ++ifr)
      emission_spec(ifr) =  BlackBodySpec(nu_grid(ifr), nu_grid(ifr+1));

      emission_spec(nfreq-1) = 1.0 - FitBlackBody(nu_grid(nfreq-1));

  }
  
}// end frequency grid



// return the integral (15/pi^4)\int_0^{nu/T} x^3 dx/(exp(x)-1)
// frequency is scaled with kT_0/h
// using fitting formula to return \int_0^nu_min and \int_0^nu_max

Real Radiation::FitBlackBody(Real nu_t)
{

// the integral at nu_t=1.5 is 0.6154949828394710
  Real integral = 0.0;
  Real nu_2 = nu_t * nu_t;
  Real nu_3 = nu_t * nu_2; 
  Real nu_7 = nu_2 * nu_2 * nu_3;
  if(nu_t < 1.8){
    integral = 0.051329911273422 * nu_3 -0.019248716727533 * nu_t * nu_3
               + 0.002566495563671 * nu_2 * nu_3
               -3.055351861513195*1.e-5*nu_7;
  }else if(nu_t < 18.6){
    Real exp_nu = exp(-nu_t);
    integral = -0.156915538762850 * exp_nu * (nu_3 + 2.69 * nu_2 + 6.714 * nu_t)
               + 1.000009331428801*(1- exp_nu);
  }else if(nu_t < 45){
    integral = 1.0 - 192.1 * exp(-0.9014*nu_t);
  }else{
    integral = 1.0;
  }

  return integral;
}


Real Radiation::BlackBodySpec(Real nu_min, Real nu_max)
{
  
  return (FitBlackBody(nu_max) - FitBlackBody(nu_min));

}

// In the last frequency bin, [nu, infty]
// we assume the spectrum is blackbody with effective temeprature Tr
// so that intensity=T_r^4 (15/pi^4) int_{nu/T_r}^{infty} x^3dx/(exp(x)-1)
// This is rearranged to intensity/nu^4=A=(1/y)^4(15/pi^4)\int_y^{infty} x^3dx/(exp(x)-1)
// we use a fitting formula to get y
Real Radiation::EffectiveBlackBody(Real intensity, Real nu)
{
  
  Real a_nu = intensity/(nu*nu*nu*nu); // I/nu^4
  Real nu_tr = 1.0;
  if(a_nu > 0.5){
    nu_tr = pow((1.0/a_nu),0.25);
  }else{
    Real loganu = -log(a_nu);
    nu_tr = -0.000525 * loganu * loganu * loganu + 0.03138 * loganu * loganu 
            + 0.3223 * loganu + 0.8278;
  }

  return nu_tr;

}

// In the last frequency bin, [nu, infty]
// we assum, n(nu)=1/(exp(nu/T)-1), the input value 
// n_nu2 = \int_{nu_f}^{infty} n\nu^2 d\nu
// we fit the formula n_nu2/nu^3=A=(1/y)^3\int_y^{infty} x^2/(exp(x)-1) dx

Real Radiation::EffectiveBlackBodyNNu2(Real n_nu2, Real nu)
{
  Real fit_a = n_nu2/(nu*nu*nu);
  Real log_fit_a=log(fit_a);
  Real nu_tr = 1.0;
  if(fit_a < 0.001){
    nu_tr = 0.001177*log_fit_a*log_fit_a-0.8812*log_fit_a-0.7428;
  }else if(fit_a < 5.0){
    Real exp_nu = (log_fit_a+13.28)/9.551;
    nu_tr=8.647*exp(-exp_nu*exp_nu);
  }else{
    nu_tr=pow(2.404113806319301/fit_a,1.0/3.0);
  }

  return nu_tr;

}

// return the integral (15/pi^4)\int_{\nu/T}^{\infty} \nu J_nu d\nu
// =(15/pi^4)\int_{\nu/T}^{\infty} x^4/(exp(x)-1) dx
// the input is nu_t=nu_f/T
Real Radiation::IntegrateBBNuJ(Real nu_t)
{
  Real nu_sq = nu_t*nu_t;
  Real nu_four = nu_sq * nu_sq;
  Real nu_three = nu_t * nu_sq;
  Real nu_five = nu_t * nu_four;
  Real nu_six = nu_sq * nu_four;
  Real exp_nu = exp(-nu_t);
  Real integral = 0.0;
  if(nu_t < 2.596){
    integral = 3.832229496128511 
               - 3.75 * ONE_PI_FOUR_POWER * nu_four 
               + 1.5 * ONE_PI_FOUR_POWER * nu_five 
               - (5.0/24.0) * ONE_PI_FOUR_POWER * nu_six;
  }else{
    integral = 15.0*ONE_PI_FOUR_POWER*exp_nu*(24.0
               +24.0*nu_t+12.0*nu_sq+4.0*nu_three+nu_four);
  }

  return integral;

}

// return the integral (15/pi^4)\int_{\nu/T}^{\infty} (J_nu/nu)^2 d\nu
// =(15/pi^4)\int_{\nu/T}^{\infty} x^4/(exp(x)-1)^2 dx
// the input is nu_t=nu_f/T
Real Radiation::IntegrateBBJONuSq(Real nu_t)
{
  Real nu_sq = nu_t*nu_t;
  Real nu_four = nu_sq * nu_sq;
  Real nu_three = nu_t * nu_sq;
  Real exp_nu = exp(-2.0*nu_t);
  Real integral = 0.0;
  if(nu_t < 10.0){
    Real top = 0.01872 * nu_sq - 0.2732 * nu_t + 0.9735;
    Real bottom = nu_sq - 1.854 * nu_t + 5.828;
    integral = top/bottom;

  }else{
    integral = 3.75*ONE_PI_FOUR_POWER*exp_nu*(3.0
               +6.0*nu_t+6.0*nu_sq+4.0*nu_three+2.0*nu_four);
  }

  return integral;

}

// return the integral \int_{nu/T}^{\infty} n \nu^2 d\nu
// = \int_{nu/T}^{\infty} x^2/(exp(x)-1) dx
// the input is nu_t=nu_f/T_r
Real Radiation::IntegrateBBNNu2(Real nu_t)
{
  Real nu_sq = nu_t*nu_t;
  Real nu_three = nu_t * nu_sq;
  Real nu_four = nu_sq * nu_sq;
  Real nu_six = nu_three * nu_three;
  Real nu_eight = nu_four * nu_four;
  Real exp_nu = exp(-nu_t);
  Real integral = 0.0;
  if(nu_t < 3.297){
    integral = 2.404113806319301 - 0.5 * nu_sq + (1.0/6.0) * nu_three
             - (1.0/48.0) * nu_four + (1.0/4320.0) * nu_six 
             - (1.0/241920.0) * nu_eight;
  }else{
    integral = exp_nu * (2.0 + 2.0 * nu_t + nu_sq);
  }

  return integral;
}

