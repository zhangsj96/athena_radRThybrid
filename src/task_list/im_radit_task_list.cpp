//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft_grav_task_list.cpp
//! \brief function implementation for FFTGravitySolverTaskList

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../mesh/mesh.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/implicit/radiation_implicit.hpp"
#include "../radiation/integrators/rad_integrators.hpp"
#include "../hydro/hydro.hpp"
#include "im_rad_task_list.hpp"

//----------------------------------------------------------------------------------------
//! IMRadTaskList constructor

IMRadITTaskList::IMRadITTaskList(Mesh *pm) {
  pmy_mesh = pm;
  // Now assemble list of tasks for each stage of time integrator
  {using namespace IMRadITTaskNames; // NOLINT (build/namespace)
    // compute hydro fluxes, integrate hydro variables
    AddTask(ADD_FLX_DIV,NONE);
    AddTask(ADD_ANG_FLX,ADD_FLX_DIV);
    AddTask(CAL_RAD_SCR,ADD_ANG_FLX);    
    AddTask(SEND_RAD_BND,CAL_RAD_SCR);
    AddTask(RECV_RAD_BND,CAL_RAD_SCR);
    AddTask(SETB_RAD_BND,(RECV_RAD_BND|SEND_RAD_BND));
    if(pm->shear_periodic){
      AddTask(SEND_RAD_SH,SETB_RAD_BND);
      AddTask(RECV_RAD_SH,SEND_RAD_SH|RECV_RAD_BND);
    }
    TaskID setb = SETB_RAD_BND;
    if(pm->shear_periodic)
      setb=(setb|RECV_RAD_SH);
    if(pm->multilevel){
      AddTask(PRLN_RAD_BND,setb);
      AddTask(RAD_PHYS_BND,PRLN_RAD_BND);
    }else{
      AddTask(RAD_PHYS_BND,setb);
    }
    AddTask(CLEAR_RAD, RAD_PHYS_BND);
    // check residual does not need ghost zones
    AddTask(CHK_RAD_RES,NONE);
  } // end of using namespace block

}

//----------------------------------------------------------------------------------------
//! \fn void FFTGravitySolverTaskList::AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void IMRadITTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;

  using namespace IMRadITTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_RAD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::ClearRadBoundary);
  } else if (id == SEND_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::SendRadBoundary);
  } else if (id == RECV_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::ReceiveRadBoundary);
  } else if (id == SETB_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::SetRadBoundary);
  } else if (id == RAD_PHYS_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::PhysicalBoundary);
   }else if (id == PRLN_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::ProlongateBoundary);
  } else if (id == SEND_RAD_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::SendRadBoundaryShear);
  } else if (id == RECV_RAD_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::ReceiveRadBoundaryShear);
  } else if (id == CHK_RAD_RES) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::CheckResidual);
  } else if (id == ADD_FLX_DIV) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::AddFluxDivergence);
  } else if (id == ADD_ANG_FLX) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::AddAngularFlux);
  } else if (id == CAL_RAD_SCR) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadITTaskList::CalSourceTerms);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in IMRadTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}

TaskStatus IMRadITTaskList::AddFluxDivergence(MeshBlock *pmb) {
  int &ite_scheme = pmy_mesh->pimrad->ite_scheme;
  Radiation *prad = pmb->prad;
  if(ite_scheme == 0 || ite_scheme == 2)
    prad->pradintegrator->FirstOrderFluxDivergence(prad->ir);
  else if(ite_scheme == 1)
    prad->pradintegrator->FirstOrderGSFluxDivergence(dt, prad->ir);  
  else{
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Iteration]"
        << std::endl << "ite_scheme_ '" << ite_scheme << "' not allowed!";
    ATHENA_ERROR(msg);
  }
  return TaskStatus::success;
  
}

TaskStatus IMRadITTaskList::AddAngularFlux(MeshBlock *pmb) {

  Radiation *prad = pmb->prad;
  if(prad->angle_flag == 1){
    prad->pradintegrator->ImplicitAngularFluxes(prad->ir);  
  }

  return TaskStatus::success;
  
}


TaskStatus IMRadITTaskList::CalSourceTerms(MeshBlock *pmb) {

  Radiation *prad = pmb->prad;
  Hydro *ph = pmb->phydro;
  
  prad->pradintegrator->CalSourceTerms(pmb, dt, ph->u, prad->ir1, prad->ir);


  return TaskStatus::success;
  
}



TaskStatus IMRadITTaskList::ClearRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.ClearBoundary(BoundaryCommSubset::radiation);
  return TaskStatus::success;
}

TaskStatus IMRadITTaskList::SendRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

TaskStatus IMRadITTaskList::SendRadBoundaryShear(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SendShearingBoxBoundaryBuffers();
  return TaskStatus::success;
}


TaskStatus IMRadITTaskList::ReceiveRadBoundary(MeshBlock *pmb) {
  bool ret = pmb->prad->rad_bvar.ReceiveBoundaryBuffers();
  if (!ret)
    return TaskStatus::fail;
  return TaskStatus::success;
}

TaskStatus IMRadITTaskList::ReceiveRadBoundaryShear(MeshBlock *pmb) {
  bool ret = pmb->prad->rad_bvar.ReceiveShearingBoxBoundaryBuffers();
  if(ret){
    pmb->prad->rad_bvar.SetShearingBoxBoundaryBuffers();
    return TaskStatus::success;
  }else{
    return TaskStatus::fail;
  }
}


TaskStatus IMRadITTaskList::SetRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SetBoundaries();
  return TaskStatus::success;
}


TaskStatus IMRadITTaskList::CheckResidual(MeshBlock *pmb) {
  pmy_mesh->pimrad->CheckResidual(pmb, pmb->prad->ir_old,pmb->prad->ir);
  return TaskStatus::success;
}

void IMRadITTaskList::StartupTaskList(MeshBlock *pmb) {
  pmb->prad->rad_bvar.StartReceiving(BoundaryCommSubset::radiation);
  if(pmy_mesh->shear_periodic)
    pmb->prad->rad_bvar.StartReceivingShear(BoundaryCommSubset::radiation);
  return;
}


