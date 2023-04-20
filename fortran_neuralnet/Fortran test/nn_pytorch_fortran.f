! This is a program that reads a traced neural network and does an inference with it. Input is written inside this file
program forward
  
  ! Import the files
  use torch_ftn
  use iso_fortran_env
  
  implicit none

  ! Define variables etc
  integer :: n
  type(torch_module) :: torch_mod
  type(torch_tensor_wrap) :: input_tensors
  type(torch_tensor) :: out_tensor

  ! Specify input and output variables
  real(real32) :: input(224, 224, 3, 10)
  real(real32), pointer :: output(:, :)

  ! Allocate filename and others
  character(:), allocatable :: filename
  integer :: arglen, stat

  ! Check if the provided argument is a single argument
  if (command_argument_count() /= 1) then
    print *, "Need to pass a single argument: Pytorch model file name"
    ! Stop if this was not the case
    stop
  end if

  ! Read the length of the specified model
  call get_command_argument(number=1, length=arglen)
  ! With this length, allocate string of correct length 
  allocate(character(arglen) :: filename)
  ! Now, save the string into the variable filename to read the PyTorch model
  call get_command_argument(number=1, value=filename, status=stat)
  
  ! NOTE : the input is written here, this should be changed to take an input argument!
  input = 1.0
  call input_tensors%create
  call input_tensors%add_array(input)
  call torch_mod%load(filename)
  call torch_mod%forward(input_tensors, out_tensor)
  call out_tensor%to_array(output)

  print *, output(1:5,1)
end program
