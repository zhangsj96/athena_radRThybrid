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
#include "../../eos/eos.hpp"

RadIntegrator::RadIntegrator(Radiation *prad, ParameterInput *pin)
{

  pmy_rad = prad;

  MeshBlock *pmb = prad->pmy_block;
  Coordinates *pco=pmb->pcoord;

  int nang=prad->nang;
  int nfreq=prad->nfreq;
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
  taufact_ = pin->GetOrAddReal("radiation","taucell",1);
  tau_flag_ = pin->GetOrAddInteger("radiation","tau_scheme",1);
  compton_flag_=pin->GetOrAddInteger("radiation","Compton",0);
  planck_flag_=pin->GetOrAddInteger("radiation","Planck",0);
  adv_flag_=pin->GetOrAddInteger("radiation","Advection",1);
  flux_correct_flag_ = pin->GetOrAddInteger("radiation","CorrectFlux",0);




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

  sfac1_x_.NewAthenaArray(ncells1,prad->n_fre_ang);
  sfac2_x_.NewAthenaArray(ncells1,prad->n_fre_ang);  
  if(ncells2 > 1){
    sfac1_y_.NewAthenaArray(ncells2,ncells1,prad->n_fre_ang);
    sfac2_y_.NewAthenaArray(ncells2,ncells1,prad->n_fre_ang);    
  }
  if(ncells3 > 1){
    sfac1_z_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    sfac2_z_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);    
  }

  if(adv_flag_ > 0){
    adv_vel.NewAthenaArray(3,ncells3,ncells2,ncells1);
  }
  
  if(IM_RADIATION_ENABLED){
    limiter_.NewAthenaArray(ncells1,prad->n_fre_ang);

    if(ncells2 > 1){
      limiterj_.NewAthenaArray(ncells2,ncells1,prad->n_fre_ang);

    }
    if(ncells3 > 1){
      limiterk_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);

    }
    dql_.NewAthenaArray(prad->n_fre_ang);
    dqr_.NewAthenaArray(prad->n_fre_ang);

    const_coef1_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    if(ncells2 > 1){
      const_coef2_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    }
    if(ncells3 > 1){
      const_coef3_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    }
    divflx_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);


  
    left_coef1_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    left_coef2_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    left_coef3_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);

    p_velx_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    n_velx_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    p_vely_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    n_vely_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    p_velz_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
    n_velz_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);

    ang_flx_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);


  }// end implicit
  implicit_coef_.NewAthenaArray(prad->n_fre_ang);

  tgas_.NewAthenaArray(ncells3,ncells2,ncells1);
  vel_source_.NewAthenaArray(ncells3,ncells2,ncells1,3); 


  vel_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
  velx_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
  vely_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
  velz_.NewAthenaArray(ncells3,ncells2,ncells1,prad->n_fre_ang);
  
  vncsigma_.NewAthenaArray(nang);
  vncsigma2_.NewAthenaArray(nang);
  wmu_cm_.NewAthenaArray(nang);
  tran_coef_.NewAthenaArray(nang);
  cm_to_lab_.NewAthenaArray(nang);
  ir_cm_.NewAthenaArray(prad->n_fre_ang);



  if(prad->angle_flag == 1){
    int &nzeta = prad->nzeta;
    int &npsi = prad->npsi;
    if(nzeta > 0){
      g_zeta_.NewAthenaArray(2*nzeta+1);
      q_zeta_.NewAthenaArray(2*nzeta+2*NGHOST);
      ql_zeta_.NewAthenaArray(2*nzeta+2*NGHOST);
      qr_zeta_.NewAthenaArray(2*nzeta+2*NGHOST);
      

      if(npsi > 0){
        zeta_flux_.NewAthenaArray(ncells3,ncells2,ncells1,(2*nzeta+1)*2*npsi);
        zeta_area_.NewAthenaArray(2*npsi,2*nzeta+1);
      }
      else{
        zeta_flux_.NewAthenaArray(ncells3,ncells2,ncells1,2*nzeta+1);
        zeta_area_.NewAthenaArray(2*nzeta+1);
      }

      pco->ZetaArea(prad, zeta_area_);

    }

    if(npsi > 0){
      g_psi_.NewAthenaArray(2*npsi+1);
      q_psi_.NewAthenaArray(2*npsi+2*NGHOST);
      ql_psi_.NewAthenaArray(2*npsi+2*NGHOST);
      qr_psi_.NewAthenaArray(2*npsi+2*NGHOST);      


      if(nzeta > 0){
        psi_flux_.NewAthenaArray(ncells3,ncells2,ncells1,2*nzeta*(2*npsi+1));
        psi_area_.NewAthenaArray(2*nzeta,2*npsi+1);
      }
      else{
        psi_flux_.NewAthenaArray(ncells3,ncells2,ncells1,2*npsi+1); 
        psi_area_.NewAthenaArray(2*npsi+1);
      }

      pco->PsiArea(prad, psi_area_); 
    }

    dflx_ang_.NewAthenaArray(nang);
    ang_vol_.NewAthenaArray(nang);
    pco->AngularVol(prad, ang_vol_);
  }

  // calculate the advection velocity at the cell faces

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int il, iu, jl, ju, kl, ku;
  jl = js, ju=je, kl=ks, ku=ke;

  if(ncells2 > 1)
  {
    if(ncells3 == 1){
      jl=js-1, ju=je+1, kl=ks, ku=ke;
    }else{
      jl=js-1, ju=je+1, kl=ks-1, ku=ke+1;
    }
 
  }

  // calculate velx_
  for (int k=kl; k<=ku; ++k){
    for (int j=jl; j<=ju; ++j){
        // get the velocity at the interface
      for(int i=is-1; i<=ie+1; ++i){
        Real dxl = pco->x1f(i)-pco->x1v(i-1);
        Real dxr = pco->x1v(i) - pco->x1f(i);
        Real factl = dxr/(dxl+dxr);
        Real factr = dxl/(dxl+dxr);
        for(int ifr=0; ifr<nfreq; ++ifr){
          Real *cosx = &(prad->mu(0,k,j,i-1,0));
          Real *cosx1 = &(prad->mu(0,k,j,i,0));
          Real *veln = &(velx_(k,j,i,ifr*nang));
#pragma omp simd aligned(cosx,cosx1,veln:ALI_LEN)
          for(int n=0; n<nang; ++n){
            // linear intepolation between x1v(i-1), x1f(i), x1v(i)
            veln[n] = prad->reduced_c *
                                (factl * cosx[n] + factr * cosx1[n]);                       
          }// end n
        }// end ifr
      }// end i
    }
  }// end k

  // calculate vely_
  if(ncells2 > 1){
    il = is-1, iu = ie+1, kl = ks, ku = ke;
    if (ncells3 >  1) // 2D
      kl = ks-1, ku = ke+1;

    for (int k=kl; k<=ku; ++k){
      for (int j=js; j<=je+1; ++j){
        // get the velocity at the interface
        for(int i=il; i<=iu; ++i){
          Real dxl = pco->x2f(j)-pco->x2v(j-1);
          Real dxr = pco->x2v(j) - pco->x2f(j);
          Real factl = dxr/(dxl+dxr);
          Real factr = dxl/(dxl+dxr);
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real *cosy = &(prad->mu(1,k,j-1,i,0));
            Real *cosy1 = &(prad->mu(1,k,j,i,0));
            Real *veln = &(vely_(k,j,i,ifr*nang));
#pragma omp simd aligned(cosy,cosy1,veln:ALI_LEN)
            for(int n=0; n<nang; ++n){
            // linear intepolation between x2v(j-1), x2f(j), x2v(j)
              veln[n] = prad->reduced_c *
                          (factl * cosy[n] + factr * cosy1[n]);
            }


          }// end ifr
        }// end i
      }
    }// end k 
  }// ncells2

  // calculate vely_
  if(ncells3 > 1){
    il =is-1, iu=ie+1, jl=js-1, ju=je+1;

    for (int k=ks; k<=ke+1; ++k){
      for (int j=jl; j<=ju; ++j){
        // get the velocity at the interface
        for(int i=il; i<=iu; ++i){
          Real dxl = pco->x3f(k) - pco->x3v(k-1);
          Real dxr = pco->x3v(k) - pco->x3f(k);
          Real factl = dxr/(dxl+dxr);
          Real factr = dxl/(dxl+dxr);
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real *cosz = &(prad->mu(2,k-1,j,i,0));
            Real *cosz1 = &(prad->mu(2,k,j,i,0));
            Real *veln = &(velz_(k,j,i,ifr*nang));
#pragma omp simd aligned(cosz,cosz1,veln:ALI_LEN)
            for(int n=0; n<nang; ++n){
            // linear intepolation between x2v(j-1), x2f(j), x2v(j)
              veln[n] = prad->reduced_c *
                          (factl * cosz[n] + factr * cosz1[n]);
            }
          }// end ifr
        }// end i
      }// end j
    }// end k 
  }// ncells3

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
  
  vel_.DeleteAthenaArray();
  velx_.DeleteAthenaArray();
  vely_.DeleteAthenaArray();
  velz_.DeleteAthenaArray();

  
  vncsigma_.DeleteAthenaArray();
  vncsigma2_.DeleteAthenaArray();
  wmu_cm_.DeleteAthenaArray();
  tran_coef_.DeleteAthenaArray();
  cm_to_lab_.DeleteAthenaArray();
  ir_cm_.DeleteAthenaArray();

  sfac1_x_.DeleteAthenaArray();
  sfac2_x_.DeleteAthenaArray();  
  if(pmy_rad->pmy_block->ncells2 > 1){
    sfac1_y_.DeleteAthenaArray();
    sfac2_y_.DeleteAthenaArray();    
  }
  if(pmy_rad->pmy_block->ncells3 > 1){
    sfac1_z_.DeleteAthenaArray();
    sfac2_z_.DeleteAthenaArray();    
  }

  if(adv_flag_ > 0){
    adv_vel.DeleteAthenaArray();
  }

  if(IM_RADIATION_ENABLED){

    const_coef1_.DeleteAthenaArray();
    limiter_.DeleteAthenaArray();

    if(pmy_rad->pmy_block->ncells2 > 1){
      const_coef2_.DeleteAthenaArray();
      limiterj_.DeleteAthenaArray();
    }
    if(pmy_rad->pmy_block->ncells3 > 1){
      const_coef3_.DeleteAthenaArray();
      limiterk_.DeleteAthenaArray();
    }
    divflx_.DeleteAthenaArray();
    dql_.DeleteAthenaArray();
    dqr_.DeleteAthenaArray();

    left_coef1_.DeleteAthenaArray();
    left_coef2_.DeleteAthenaArray();
    left_coef3_.DeleteAthenaArray();

    p_velx_.DeleteAthenaArray();
    n_velx_.DeleteAthenaArray();
    p_vely_.DeleteAthenaArray();
    n_vely_.DeleteAthenaArray();
    p_velz_.DeleteAthenaArray();
    n_velz_.DeleteAthenaArray();
    ang_flx_.DeleteAthenaArray();
  }
  implicit_coef_.DeleteAthenaArray();

  tgas_.DeleteAthenaArray();
  vel_source_.DeleteAthenaArray();


  if(pmy_rad->angle_flag == 1){
    int &nzeta = pmy_rad->nzeta;
    int &npsi = pmy_rad->npsi;
    if(nzeta > 0){
      g_zeta_.DeleteAthenaArray();
      q_zeta_.DeleteAthenaArray();
      ql_zeta_.DeleteAthenaArray();
      qr_zeta_.DeleteAthenaArray();
      zeta_flux_.DeleteAthenaArray();
      zeta_area_.DeleteAthenaArray();
    }
    if(npsi > 0){
      g_psi_.DeleteAthenaArray();
      q_psi_.DeleteAthenaArray();
      ql_psi_.DeleteAthenaArray();
      qr_psi_.DeleteAthenaArray();
      psi_flux_.DeleteAthenaArray();     
      psi_area_.DeleteAthenaArray(); 
    }
    dflx_ang_.DeleteAthenaArray();
    ang_vol_.DeleteAthenaArray();
  }  
  
}



void RadIntegrator::GetTgasVel(MeshBlock *pmb, const Real dt,
    AthenaArray<Real> &u, AthenaArray<Real> &w, 
    AthenaArray<Real> &bcc, AthenaArray<Real> &ir)
{


  Real gm1 = pmb->peos->GetGamma() - 1.0;

  Real rho_floor = pmb->peos->GetDensityFloor();

  Radiation *prad=pmb->prad;  
  Coordinates *pco=pmb->pcoord;

  Real& prat = prad->prat;
  Real invcrat = 1.0/prad->crat;

  int &nang =prad->nang;
  int &nfreq=prad->nfreq;


  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
 
  for(int k=0; k<pmb->ncells3; ++k){
    for(int j=0; j<pmb->ncells2; ++j){
      for(int i=0; i<pmb->ncells1; ++i){

        // for implicit update, using the quantities from the partially 
        // updated u, not from w 

         Real rho = u(IDN,k,j,i);
         rho = std::max(rho,rho_floor);
         Real vx = u(IM1,k,j,i)/rho;
         Real vy = u(IM2,k,j,i)/rho;
         Real vz = u(IM3,k,j,i)/rho;
         Real pb = 0.0;
         if(MAGNETIC_FIELDS_ENABLED)
           pb = 0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))
                +SQR(bcc(IB3,k,j,i)));

         Real vel = vx * vx + vy * vy + vz * vz;
         Real tgas = u(IEN,k,j,i) - pb - 0.5*rho*vel;
         tgas = gm1*tgas/rho;
         tgas = std::max(tgas,pmb->prad->t_floor_);
         tgas_(k,j,i) = tgas;
        // Do not use the velocity directly in strongly radiation pressure
         // dominated regime
         // use the predicted velocity based on moment equatio

         // calculate radiation energy density
         Real er = 0.0;
         for(int ifr=0; ifr<nfreq; ++ifr){ 
           Real *irn = &(ir(k,j,i,ifr*nang));
           Real *weight = &(prad->wmu(0));
           Real er_freq = 0.0;
#pragma omp simd reduction(+:er_freq)
           for(int n=0; n<nang; ++n){
             er_freq += weight[n] * irn[n];
           }
           er_freq *= prad->wfreq(ifr);
           er += er_freq;
         }   

         // now the velocity term, 
         // using velocity from current stage
         vx = w(IVX,k,j,i);
         vy = w(IVY,k,j,i);
         vz = w(IVZ,k,j,i);

         vel = vx * vx + vy * vy + vz * vz;

         if(prat * er * invcrat * invcrat > rho){

            PredictVel(ir,k,j,i, 0.5 * dt, rho, &vx, &vy, &vz);
            vel = vx * vx + vy * vy + vz * vz;

         }
        
         Real ratio = sqrt(vel) * invcrat;
         // Limit the velocity to be smaller than the speed of light
         if(ratio > prad->vmax){
           Real factor = prad->vmax/ratio;
           vx *= factor;
           vy *= factor;
           vz *= factor;
           
         }
         vel_source_(k,j,i,0) = vx;
         vel_source_(k,j,i,1) = vy;
         vel_source_(k,j,i,2) = vz;


   
      }// end i
    }// end j
  }// end k  

  // Now get interface velocity
  if(adv_flag_ > 0){
    // vx
    for(int k=ks; k<=ke; ++k){
      for(int j=js; j<=je; ++j){
        for(int i=is; i<=ie+1; ++i){
          Real tau = 0.0;
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real sigmal = prad->sigma_a(k,j,i-1,ifr) + prad->sigma_s(k,j,i-1,ifr);
            Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
            tau += prad->wfreq(ifr)*((pco->x1f(i) - pco->x1v(i-1)) * sigmal 
                    + (pco->x1v(i) - pco->x1f(i)) * sigmar);
          }// end ifr

          Real factor = 0.0;
          GetTaufactorAdv(tau,factor);
          Real vl = vel_source_(k,j,i-1,0);
          Real vr = vel_source_(k,j,i,0);
          adv_vel(0,k,j,i) = factor*(vl + (pco->x1f(i) - pco->x1v(i-1)) * 
                                 (vr - vl)/(pco->x1v(i) - pco->x1v(i-1)));

        }
      }
    }
    if(je > js){
      for(int k=ks; k<=ke; ++k){
        for(int j=js; j<=je+1; ++j){
          Real ratio = (pco->x2f(j) - pco->x2v(j-1))/
                       (pco->x2v(j) - pco->x2v(j-1));
          for(int i=is; i<=ie; ++i){
            Real tau = 0.0;
            for(int ifr=0; ifr<nfreq; ++ifr){
              Real sigmal = prad->sigma_a(k,j-1,i,ifr) + prad->sigma_s(k,j-1,i,ifr);
              Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
              tau += prad->wfreq(ifr)*((pco->x2f(j) - pco->x2v(j-1)) * sigmal 
                    + (pco->x2v(j) - pco->x2f(j)) * sigmar);
            }// end ifr

            Real factor = 0.0;
            GetTaufactorAdv(tau,factor);

            Real vl = vel_source_(k,j-1,i,1);
            Real vr = vel_source_(k,j,i,1);
            adv_vel(1,k,j,i) = factor*(vl +  ratio * (vr - vl));
          }
        }// end j
      }// end k

    }// end je > js

    if(ke > ks){
      for(int k=ks; k<=ke+1; ++k){
        Real ratio = (pco->x3f(k) - pco->x3v(k-1))/
                     (pco->x3v(k) - pco->x3v(k-1));
        for(int j=js; j<=je; ++j){
          for(int i=is; i<=ie; ++i){
            Real tau = 0.0;
            for(int ifr=0; ifr<nfreq; ++ifr){
              Real sigmal = prad->sigma_a(k-1,j,i,ifr) + prad->sigma_s(k-1,j,i,ifr);
              Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
              tau += prad->wfreq(ifr)*((pco->x3f(k) - pco->x3v(k-1)) * sigmal 
                    + (pco->x3v(j) - pco->x3f(j)) * sigmar);
            }// end ifr

            Real factor = 0.0;
            GetTaufactorAdv(tau,factor);

            Real vl = vel_source_(k-1,j,i,2);
            Real vr = vel_source_(k,j,i,2);
            adv_vel(2,k,j,i) = factor * (vl +  ratio * (vr - vl));
          }
        }// end j
      }// end k

    }// end ke > ks

  }// end adv_flag

}// end the function



