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
//! \file absorption.cpp
//  \brief  Add absorption source terms
//======================================================================================

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../radiation.hpp"
#include "../../mesh/mesh.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../utils/utils.hpp"

// this class header
#include "./rad_integrators.hpp"


//--------------------------------------------------------------------------------------
//! \fn RadIntegrator::ComToLab(const Real vx, const Real vy, const Real vz,
//                          AthenaArray<Real> &ir, AthenaArray<Real> &ir_cm)
//  \brief Return the integral of blackbody spectrum

// return the integral (15/pi^4)\int_0^{nu/T} x^3 dx/(exp(x)-1)
// frequency is scaled with kT_0/h
// using fitting formula to return \int_0^nu_min and \int_0^nu_max

Real RadIntegrator::FitBlackBody(Real nu_t)
{

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


Real RadIntegrator::BlackBodySpec(Real nu_min, Real nu_max)
{
  
  return (FitBlackBody(nu_max) - FitBlackBody(nu_min));

}


