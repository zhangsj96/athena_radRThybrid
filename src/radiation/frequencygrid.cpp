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


  if(fre_ratio > 1){

    if(nu_min < TINY_NUMBER){
      std::stringstream msg;
      msg << "### FATAL ERROR in Radiation Class" << std::endl
          << "frequency_min needs to be larger than 0!";
      throw std::runtime_error(msg.str().c_str());
    }

    if(nu_max <= nu_min){
      std::stringstream msg;
      msg << "### FATAL ERROR in Radiation Class" << std::endl
          << "frequency_max needs to be larger than frequency_min!";
      throw std::runtime_error(msg.str().c_str());
    }


    if(nfreq > 1){
      nu_max = nu_min * pow(fre_ratio,nfreq-1);
    }else{
      nfreq = log10(nu_max/nu_min)/log10(fre_ratio)+1;
    }// calculate nfreq if not given

  }else{
    if(nfreq > 2){
      if(nu_max <= nu_min){
        std::stringstream msg;
        msg << "### FATAL ERROR in Radiation Class" << std::endl
            << "frequency_max needs to be larger than frequency_min!";
        throw std::runtime_error(msg.str().c_str());
      }

      fre_ratio = log10(nu_max/nu_min)/(nfreq - 1);
      fre_ratio = pow(10.0,fre_ratio);
    }// end nfreq > 1
  }

  if(nfreq > 1){
    nu_grid.NewAthenaArray(nfreq);
    nu_grid(0) = 0.0;
    nu_grid(1) = nu_min;

    if(nfreq > 2){
      for(int n=2; n<nfreq; ++n)
      nu_grid(n) = nu_grid(n-1) * fre_ratio;
    }// end nfreq > 2

  }
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
  if(nu_t < 1.5){
    integral = 0.051329911273422 * nu_3 -0.019248716727533 * nu_t * nu_3
               + 0.002566495563671 * nu_2 * nu_3;
  }else{
    Real exp_nu = exp(-nu_t);
    integral = -0.156915538762850 * exp_nu * (nu_3 + 2.69 * nu_2 + 6.714 * nu_t)
               + 1.000009331428801*(1- exp_nu);
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


