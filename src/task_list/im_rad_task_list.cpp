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
#include "im_rad_task_list.hpp"

//----------------------------------------------------------------------------------------
//! IMRadTaskList constructor

IMRadTaskList::IMRadTaskList(): ntasks(0),  task_list_{}  {
  // Now assemble list of tasks for each stage of time integrator
  {using namespace IMRadTaskNames; // NOLINT (build/namespace)
    // compute hydro fluxes, integrate hydro variables
    AddTask(SEND_RAD_BND,NONE);
    AddTask(RECV_RAD_BND,NONE);
    AddTask(SETB_RAD_BND,(RECV_RAD_BND|SEND_RAD_BND));
    AddTask(RAD_PHYS_BND,SETB_RAD_BND);
    AddTask(CLEAR_RAD, RAD_PHYS_BND);
  } // end of using namespace block
}

//----------------------------------------------------------------------------------------
//! \fn void FFTGravitySolverTaskList::AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void IMRadTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;

  using namespace IMRadTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_RAD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*,Real, Real)>
        (&IMRadTaskList::ClearRadBoundary);
  } else if (id == SEND_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*,Real, Real)>
        (&IMRadTaskList::SendRadBoundary);
  } else if (id == RECV_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*,Real, Real)>
        (&IMRadTaskList::ReceiveRadBoundary);
  } else if (id == SETB_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*,Real,Real)>
        (&IMRadTaskList::SetRadBoundary);
  } else if (id == RAD_PHYS_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*,Real,Real)>
        (&IMRadTaskList::PhysicalBoundary);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in IMRadTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}

void IMRadTaskList::StartupTaskList(MeshBlock *pmb, Real t_end, Real wght) {
  pmb->prad->rad_bvar.StartReceiving(BoundaryCommSubset::radiation);
  return;
}

TaskStatus IMRadTaskList::ClearRadBoundary(MeshBlock *pmb, Real t_end, Real wght) {
  pmb->prad->rad_bvar.ClearBoundary(BoundaryCommSubset::radiation);
  return TaskStatus::success;
}

TaskStatus IMRadTaskList::SendRadBoundary(MeshBlock *pmb, Real t_end, Real wght) {
  pmb->prad->rad_bvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

TaskStatus IMRadTaskList::ReceiveRadBoundary(MeshBlock *pmb, Real t_end, Real wght) {
  bool ret = pmb->prad->rad_bvar.ReceiveBoundaryBuffers();
  if (!ret)
    return TaskStatus::fail;
  return TaskStatus::success;
}

TaskStatus IMRadTaskList::SetRadBoundary(MeshBlock *pmb, Real t_end, Real wght) {
  pmb->prad->rad_bvar.SetBoundaries();
  return TaskStatus::success;
}

TaskStatus IMRadTaskList::PhysicalBoundary(MeshBlock *pmb, Real t_end, Real wght) {
  pmb->pbval->ApplyPhysicalBoundaries(t_end, wght,pmb->pbval->bvars_main_int);
  return TaskStatus::success;
}
