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
//! \file rad_iteration.cpp
//  \brief iterations to solve the transport equation implicitly
//======================================================================================


// Athena++ headers
#include "./radiation.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../defs.hpp"

//--------------------------------------------------------------------------------------
// \!fn void CalculateMoment()

// \brief function to create the radiation moments


// calculate the frequency integrated moments of the radiation field
// including the ghost zones



// void IMRadiation::JacobiIteration(Mesh *pm, 
//             TimeIntegratorTaskList *ptlist, int stage){
//   // perform Jacobi iteration including both source and flux terms
//   // The iteration step is: calculate flux, calculate source term, 
//   // update specific intensity, compute error
//   MeshBlock *pmb = pm->pblock;


//   Real dt = (ptlist->stage_wghts[(stage-1)].beta)*(pm->dt);

//   Real ave_wghts[3];
//   ave_wghts[0] = 1.0;
//   ave_wghts[1] = ptlist->stage_wghts[stage-1].delta;
//   ave_wghts[2] = 0.0;

//   ave_wghts[0] = ptlist->stage_wghts[stage-1].gamma_1;
//   ave_wghts[1] = ptlist->stage_wghts[stage-1].gamma_2;
//   ave_wghts[2] = ptlist->stage_wghts[stage-1].gamma_3; 

//   const Real wght = ptlist->stage_wghts[stage-1].beta*pm->dt;

//   if(stage <= nstages){

//     // go through all the mesh blocks
//     bool iteration = TRUE;
//     int niter = 0;

//     // store the initial value before the iteration
//     // this is always needed for the RHS

//     // first save initial state
//     while(pmb != nullptr){
//       Radiation *prad = pmb->prad;
//       prad->ir_ini = prad->ir;
//       prad->ir_old = prad->ir;

//       pmb = pmb->next;
//     }



//     while(iteration){
//       // initialize the pointer
//       pmb = pm->pblock;

//       sum_full_ = 0.0;
//       sum_diff_ = 0.0;

//       // we need to go through all meshblocks for each iteration
//       while(pmb != nullptr){

//         Hydro *ph = pmb->phydro;
//         Radiation *prad = pmb->prad;

//         if ((stage == 1) && (integrator == "vl2")) {
//           prad->pradintegrator->CalculateFluxes(phydro->w,  prad->ir, 1);
//         } else {
//           prad->pradintegrator->CalculateFluxes(phydro->w,  prad->ir, 
//                                      prad->pradintegrator->rad_xorder);
//         }
// // for multi-level, send flux correction
//         if(pm->multilevel){
//           // send rad flux
//           prad->rad_bvar.sendFluxCorrection();
//           // receive rad flux
//           prad->rad_bvar.ReceiveFluxCorrection();

//         }
  
//   // add flux divergence

//     // This copy ir to ir1
//         pmb->WeightedAve(prad->ir1, prad->ir, prad->ir2, ave_wghts,1);


//         if(ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0)
//           prad->ir.SwapAthenaArray(prad->ir1);
//         else
//           pmb->WeightedAve(prad->ir, prad->ir1, prad->ir2, ave_wghts,1);


//         prad->pradintegrator->FluxDivergence(wght, prad->ir_ini, prad->ir); //ir is already partially updated
//     // so ir1 stores the values from the last iteration
//     // the source term
//         prad->pradintegrator->AddSourceTerms(pmb, dt, ph->u, ph->w, pf->bcc, prad->ir_ini, prad->ir);

//         prad->rad_bvar.SendBoundaryBuffers();
//         prad->rad_bavr.ReceiveandSetBoundariesWithWait();

//         // calculate residual
//         prad->pradintegrator->CheckResidual(pmb,prad,prad->ir_old,prad->ir);

//         // update boundary condition for ir_new

//         // copy the solution over
//         prad->ir_old = prad->ir;

//         sum_full_ += prad->sum_full;
//         sum_diff_ += prad->sum_diff;

//         pmb = pmb->next;  
//       }// end while pmb

//       // MPI sum across all the cores
// #ifdef MPI_PARALLEL
//       Real global_sum = 0.0;
//       Real global_diff = 0.0;
//       MPI_Allreduce(&sum_full_, &global_sum, 1, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);
//       MPI_Allreduce(&sum_diff_, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 

//       sum_full_ = global_sum;
//       sum_diff_ = global_diff;

// #endif  

//       niter++;
//       Real tot_res = sum_diff_/sum_full_;


//       if((niter > nlimit_) || tot_res < err_limit_)
//         iteration = FALSE;
//     }


//     // After iteration, 
//     // update boundary condition for hydro variables
//     pmb = pm->pblock;
//     while(pmb != nullptr){
//       pmb->phydro->hbvar.SwapHydroQuantity(pmb->phydro->u, HydroBoundaryQuantity::cons);
//       pmb->phydro->hbvar.SendBoundaryBuffers();
//       pmb->phydro->hbvar.ReceiveandSetBoundariesWithWait();

//       // convert conservative to primitive variables
//       ptlist->Primitives(pmb,stage);

//       pmb = pmb->next;
//     }



//   }// end stage


// }// end func



// void IMRadiation::CheckResidual(MeshBlock *pmb, Radiation *prad, 
//             AthenaArray<Real> &ir_old, AthenaArray<Real> &ir_new)
// {
//   Radiation *prad=pmb->prad;
  
//   int &nang =prad->nang;
//   int &nfreq=prad->nfreq;


//   int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
//   int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

//   prad->sum_diff = 0.0;
//   prad->sum_full = 0.0;
//   for(int k=ks; k<=ke; ++k){
//     for(int j=js; j<=je; ++j){
//       for(int i=is; i<=ie; ++i){
//         for(int ifr=0; ifr<nfreq; ++ifr){
//           Real *iro = &(ir_old(k,j,i,ifr*nang));
//           Real *irn = &(ir_new(k,j,i,ifr*nang));
//           for(int n=0; n<nang; ++n){
//             sum_diff += abs(iro[n] - irn[n]);
//             sum_full += abs(irn[n]);
//           }
//         }// end ifr
//       }// end i
//     }// end j
//   }// end k
  
// }



