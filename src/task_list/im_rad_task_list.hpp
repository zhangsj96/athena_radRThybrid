#ifndef TASK_LIST_IM_RAD_LIST_HPP_
#define TASK_LIST_IM_RAD_LIST_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file im_rad_task_list.hpp
//! \brief define im_rad_task_list class
//! \brief This is used to handle the boundary condition

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "./task_list.hpp"

// forward declarations
class Mesh;
class MeshBlock;
class Radiation;
class IMRadTaskList;
class TaskID;


//----------------------------------------------------------------------------------------
//! \struct IMRadTask
//! \brief data and function pointer for an individual IMRadTask

struct IMRadTask {
  TaskID task_id;      //!> encodes task using bit positions in IMRadTaskNames
  TaskID dependency;   //!> encodes dependencies to other tasks using IMRadTaskNames
  TaskStatus (IMRadTaskList::*TaskFunc)(MeshBlock *, Real t_end, Real wght); //!> ptr to a task
};


//----------------------------------------------------------------------------------------
//! \class MultigridTaskList
//! \brief data and function definitions for IMRadTaskList class

class IMRadTaskList {
 public:
  IMRadTaskList(); // 2x direct + zero initialization
  // rule of five:
  virtual ~IMRadTaskList() = default;

  // data
  int ntasks;     //!> number of tasks in this list

  // functions
  TaskListStatus DoAllAvailableTasks(MeshBlock *pmb, TaskStates &ts);
  void DoTaskListOneStage(MeshBlock *pmb);
  void ClearTaskList() {ntasks=0;}

  // functions
  void StartupTaskList(MeshBlock *pmb, Real t_end, Real wght);
  TaskStatus ClearRadBoundary(MeshBlock *pmb, Real t_end, Real wght);
  TaskStatus SendRadBoundary(MeshBlock *pmb, Real t_end, Real wght);
  TaskStatus ReceiveRadBoundary(MeshBlock *pmb, Real t_end, Real wght);
  TaskStatus SetRadBoundary(MeshBlock *pmb, Real t_end, Real wght);
  TaskStatus PhysicalBoundary(MeshBlock *pmb, Real t_end, Real wght);



 private:

  IMRadTask task_list_[64*TaskID::kNField_];

  void AddTask(const TaskID& id, const TaskID& dep);
};

//----------------------------------------------------------------------------------------
//! 64-bit integers with "1" in different bit positions used to ID each Multigrid task.

namespace IMRadTaskNames {
const TaskID NONE(0);
const TaskID CLEAR_RAD(1);
const TaskID SEND_RAD_BND(2);
const TaskID RECV_RAD_BND(3);
const TaskID SETB_RAD_BND(4);
const TaskID RAD_PHYS_BND(5);
} // namespace IMRadTaskNames

#endif // TASK_LIST_MG_TASK_LIST_HPP_
