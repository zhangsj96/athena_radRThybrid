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
//! \file rad_integrators.cpp
//  \brief implementation of radiation integrators
//======================================================================================

#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../radiation.hpp"
#include "rad_integrators.hpp"
#include "../../coordinates/coordinates.hpp"


RadIntegrator::RadIntegrator(Radiation *prad, ParameterInput *pin)
{

  pmy_rad = prad;

  MeshBlock *pmb = prad->pmy_block;

  rad_xorder = pin->GetOrAddInteger("time","rad_xorder",2);
  if (rad_xorder == 3) {
    if (NGHOST < 3){ 
      std::stringstream msg;
      msg << "### FATAL ERROR in radiation reconstruction constructor" << std::endl
          << "rad_xorder=" << rad_xorder <<
          " (PPM) reconstruction selected, but nghost=" << NGHOST << std::endl
          << "Reconfigure with --nghost=3  " <<std::endl;
      ATHENA_ERROR(msg);
    }
  }

  
      // factor to separate the diffusion and advection part
  taufact_ = pin->GetOrAddInteger("radiation","taucell",5);
  compton_flag_=pin->GetOrAddInteger("radiation","Compton",0);
  planck_flag_=pin->GetOrAddInteger("radiation","Planck",0);
  adv_flag_=pin->GetOrAddInteger("radiation","Advection",0);
  flux_correct_flag_ = pin->GetOrAddInteger("radiation","CorrectFlux",0);
  tau_limit_ =  pin->GetOrAddReal("radiation","tau_limit",0);



  int ncells1 = pmb->ncells1, ncells2 = pmb->ncells2, 
  ncells3 = pmb->ncells3; 

 
  
  x1face_area_.NewAthenaArray(ncells1+1);
  if(ncells2 > 1) {
    x2face_area_.NewAthenaArray(ncells1);
    x2face_area_p1_.NewAthenaArray(ncells1);
  }
  if(ncells3 > 1) {
    x3face_area_.NewAthenaArray(ncells1);
    x3face_area_p1_.NewAthenaArray(ncells1);
  }
  cell_volume_.NewAthenaArray(ncells1);


  cwidth2_.NewAthenaArray(ncells1);
  cwidth3_.NewAthenaArray(ncells1);

  dflx_.NewAthenaArray(ncells1,prad->n_fre_ang);

  // arrays for spatial recontruction 
  il_.NewAthenaArray(ncells1,prad->n_fre_ang);
  ilb_.NewAthenaArray(ncells1,prad->n_fre_ang);

  ir_.NewAthenaArray(ncells1,prad->n_fre_ang);
  
  temp_i1_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
  temp_i2_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
  
  vel_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
  vel2_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
  
  vncsigma_.NewAthenaArray(prad->nang);
  vncsigma2_.NewAthenaArray(prad->nang);
  wmu_cm_.NewAthenaArray(prad->nang);
  tran_coef_.NewAthenaArray(prad->nang);
  cm_to_lab_.NewAthenaArray(prad->nang);
  ir_cm_.NewAthenaArray(prad->n_fre_ang);



 

}
// destructor

RadIntegrator::~RadIntegrator()
{
 
  x1face_area_.DeleteAthenaArray();
  if(pmy_rad->pmy_block->ncells2 > 1) {
    x2face_area_.DeleteAthenaArray();
    x2face_area_p1_.DeleteAthenaArray();
  }
  if(pmy_rad->pmy_block->ncells3 > 1) {
    x3face_area_.DeleteAthenaArray();
    x3face_area_p1_.DeleteAthenaArray();
  }
  cell_volume_.DeleteAthenaArray();

  cwidth2_.DeleteAthenaArray();
  cwidth3_.DeleteAthenaArray();

  dflx_.DeleteAthenaArray();

  il_.DeleteAthenaArray();
  ilb_.DeleteAthenaArray();

  ir_.DeleteAthenaArray();

  temp_i1_.DeleteAthenaArray();
  temp_i2_.DeleteAthenaArray();

  vel_.DeleteAthenaArray();
  vel2_.DeleteAthenaArray();

  
  vncsigma_.DeleteAthenaArray();
  vncsigma2_.DeleteAthenaArray();
  wmu_cm_.DeleteAthenaArray();
  tran_coef_.DeleteAthenaArray();
  cm_to_lab_.DeleteAthenaArray();
  ir_cm_.DeleteAthenaArray();
  
}

