program main
  implicit none

  !> Initialization
  ! Input x
  real, dimension(3, 1) :: x
  ! Declare auxiliary matrices (weights and biases)
  real, dimension(:,:), allocatable :: stack0_weight
  real, dimension(:,:), allocatable :: stack0_bias
  real, dimension(:,:), allocatable :: stack2_weight
  real, dimension(:,:), allocatable :: stack2_bias
  real, dimension(:,:), allocatable :: stack4_weight
  real, dimension(:,:), allocatable :: stack4_bias
  ! Output y
  real, dimension(1, 1) :: y

  ! Declare other variables, such as filename and integers i, j for loops
  character(len=256) :: fname
  integer            :: i, j

  !> Interface for subroutine read matrix
  interface
    subroutine ReadMatrix(fname, nrows, ncols, matrix, un)
       implicit none

       ! Declare the variables
       character(len=256), intent(in)                  :: fname
       !character(len=256)                             :: str_line
       integer, intent(in)                             :: nrows, ncols
       integer, intent(in)                             :: un
       integer                                         :: iflag, nlines
       integer                                         :: i, j
       real, dimension(:), allocatable                 :: values
       real, dimension(:, :), allocatable, intent(out) :: matrix
       integer, dimension (1:2)                        :: order2 = (/ 2, 1 /)

    end subroutine
  end interface

  !> Read our matrices
  fname = "weight0_flat.csv"
  call ReadMatrix(fname, 600, 3, stack0_weight, 1)

  fname = "bias0_flat.csv"
  call ReadMatrix(fname, 600, 1, stack0_bias, 2)

  fname = "weight2_flat.csv"
  call ReadMatrix(fname, 200, 600, stack2_weight, 3)

  fname = "bias2_flat.csv"
  call ReadMatrix(fname, 200, 1, stack2_bias, 4)

  fname = "weight4_flat.csv"
  call ReadMatrix(fname, 1, 200, stack4_weight, 5)

  fname = "bias4_flat.csv"
  call ReadMatrix(fname, 1, 1, stack4_bias, 6)

  print *, "Reading: all done!"

  ! Make a prediction:
  x = reshape([10.204131145455385, 12.026584842282125, 22.131296926293793], [3,1])
  print *, "We are predicting for:"
  print *, x

  call predict(x, y, stack0_weight, stack0_bias, stack2_weight, stack2_bias, stack4_weight, stack4_bias)

  print *, "Prediction"
  print *, y

end program main

subroutine ReadMatrix(fname, nrows, ncols, matrix, un)
   implicit none

   ! Declare the variables
   character(len=256), intent(in)             :: fname
   character(len=256)                         :: str_line
   integer, intent(in)                        :: nrows, ncols
   integer, intent(in)                        :: un
   integer                                    :: iflag, nlines
   integer                                    :: i, j

   real, dimension(:), allocatable                :: values
   real, dimension(:, :), allocatable, intent(out) :: matrix

   integer, dimension (1:2) :: order2 = (/ 2, 1 /)

   ! Open the file for reading
   open(unit=un, file=fname, status='old')!, access='sequential')  !


   ! Allocate memory for values and matrix
   nlines = nrows*ncols
   allocate(values(nlines))
   allocate(matrix(nrows, ncols))

   do i = 0,nlines
      read(un,*,iostat=iflag) values(i)
      if (iflag/=0) exit
      !print *, i
      !print *, values(i)
   enddo

   ! Reading is over, close the file
   close(unit = un)
   matrix = reshape(values, (/ nrows, ncols /), order=order2)
end subroutine ReadMatrix


subroutine sigmoid(x, sigmoid_values)
  implicit none

  integer, dimension(2) :: s
  integer :: i
  real, dimension(:,:), intent(in)  :: x
  real, dimension(:,:), intent(out) :: sigmoid_values

  s = shape(x)

  ! Fill array with sigmoid values
  do i = 1, s(1)
      sigmoid_values(i,1) = 1 / (1 + exp(-x(i,1)))
  end do

end subroutine sigmoid


subroutine predict(x, y, W0, b0, W2, b2, W4, b4)
  implicit none

  ! Declare input and output types:
  real, intent(in)  :: x(3, 1)
  real              :: xx(600, 1)  ! intermediate result 1
  real              :: yy(600, 1)  ! intermediate result 1, with sigmoid
  real              :: xxx(200, 1) ! intermediate result 2
  real              :: yyy(200, 1) ! intermediate result 2, with sigmoid
  real, intent(out) :: y(1, 1)     ! final prediction: the pressure

  ! Declare auxiliary matrices (weights and biases), also given as input
  real, intent(in) :: W0(600, 3)
  real, intent(in) :: b0(600, 1)
  real, intent(in) :: W2(200, 600)
  real, intent(in) :: b2(200, 1)
  real, intent(in) :: W4(1, 200)
  real, intent(in) :: b4(1, 1)

  !> Interface for sigmoid function
  ! Note: we apply this to a vector, but has shape (n, 1), with n the size of the vector
  ! Vector sigmoid function
  interface
    subroutine sigmoid(x, sigmoid_values)
      implicit none

      integer, dimension(2) :: s
      integer               :: i

      real, dimension(:,:), intent(in) :: x
      real, dimension(:,:), intent(out) :: sigmoid_values

    end subroutine sigmoid
  end interface

  ! Do the calculation:
  xx  = matmul(W0, x)   + b0
  call sigmoid(xx, yy)
  xxx = matmul(W2, yy)  + b2
  call sigmoid(xxx, yyy)
  y   = matmul(W4, yyy) + b4
end subroutine predict
