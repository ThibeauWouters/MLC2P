module mod_grhd_phys_neuralnet

  !> Module that replaces the C2P conversion in GRHD with a neural network
  use mod_physics
  use mod_grhd_phys_parameters

  implicit none

  private

  !> Specify the architecture (by default, we use two hidden layers)
  !> The values hard-coded here are for the pruned neural network
  integer, parameter :: INPUT_SIZE    =      3
  integer, parameter :: HIDDEN_SIZE_1 =    504
  integer, parameter :: HIDDEN_SIZE_2 =    127
  integer, parameter :: OUTPUT_SIZE   =      1
  logical, parameter :: USE_SIGMOID   = .true. ! True for sigmoid, false for ReLU

  !> Define type that stores (information on) the weights and biases of the neural network
  type nn_table_t
     double precision, dimension(:,:), allocatable :: weight0 ! Weights first hidden layer
     double precision, dimension(:,:), allocatable :: bias0   ! Bias first hidden layer
     double precision, dimension(:,:), allocatable :: weight2 ! Weights second hidden layer
     double precision, dimension(:,:), allocatable :: bias2   ! Bias second hidden layer
     double precision, dimension(:,:), allocatable :: weight4 ! Weights output layer
     double precision, dimension(:,:), allocatable :: bias4   ! Bias output layer
  end type nn_table_t

  type(nn_table_t) :: nn_tab

  ! Public methods
  public :: grhd_phys_neuralnet_init
  public :: nn_predict

contains

  
  subroutine grhd_phys_neuralnet_init(filepath)
    !> Subroutine to initialize the module. Weights and biases should be in the same directory (filepath) and have the names shown below
    character(*), intent(in)   :: filepath 
    character(len=256)         :: fname = '' 

    !> Initialize the neural network weights and biases here by reading the CSV files
    fname = trim(adjustl(filepath))//"/weight0_flat.csv"
    call read_matrix(fname, HIDDEN_SIZE_1, INPUT_SIZE, nn_tab%weight0)
    fname = trim(adjustl(filepath))//"/bias0_flat.csv"
    call read_matrix(fname, HIDDEN_SIZE_1,           1, nn_tab%bias0)

    fname = trim(adjustl(filepath))//"/weight2_flat.csv"
    call read_matrix(fname, HIDDEN_SIZE_2, HIDDEN_SIZE_1, nn_tab%weight2)
    fname = trim(adjustl(filepath))//"/bias2_flat.csv"
    call read_matrix(fname, HIDDEN_SIZE_2,           1, nn_tab%bias2)
    
    fname = trim(adjustl(filepath))//"/weight4_flat.csv"
    call read_matrix(fname, OUTPUT_SIZE, HIDDEN_SIZE_2, nn_tab%weight4)
    fname = trim(adjustl(filepath))//"/bias4_flat.csv"
    call read_matrix(fname,   OUTPUT_SIZE,            1, nn_tab%bias4)

  end subroutine grhd_phys_neuralnet_init

  
  subroutine read_matrix(fname, nrows, ncols, matrix)
    !> Subroutine which reads the values from a CSV, loads them into a matrix, and stores them into the NN type
    implicit none

    ! Declare the variables
    character(len=256), intent(in)                              :: fname                 
    integer, intent(in)                                         :: nrows, ncols
    double precision, dimension(:, :), allocatable, intent(out) :: matrix

    ! Local variables for processing CSV files
    integer                                    :: iflag, nlines
    integer                                    :: i, j
    double precision, dimension(nrows*ncols)   :: values
    ! To reshape in the correct shape after reading flattened arrays:
    integer, dimension (1:2) :: order2 = (/ 2, 1 /)

    ! Allocate memory for matrix
    allocate(matrix(nrows, ncols))

    ! Open the file for reading
    open(unit=666, file=fname, status='old')
    do i = 1, nrows*ncols
      read(666,*,iostat=iflag) values(i)
      if (iflag/=0) exit
    end do
    ! Reading is over, close the file
    close(unit = 666)

    matrix = reshape(values, (/ nrows, ncols /), order=order2)

  end subroutine read_matrix

  
  subroutine nn_predict(D, S, tau, p)
    !> Make a prediction with the NN: Given the conserved variables D, S, tau, returns the pressure p as computed by the neural network
    implicit none

    !> Input and output of the NN
    double precision, intent(in)   :: D   ! Conservative energy density
    double precision, intent(in)   :: S   ! Conservative momentum density
    double precision, intent(in)   :: tau ! Conservative energy density relative to D
    double precision, intent(out)  :: p   ! Pressure
    double precision, dimension(3) :: x   ! Input for the neural net as a vector (D, S, tau)
    integer                        :: i, j
 
    x(1) = D
    x(2) = S
    x(3) = tau
    !> Call to make the computations:
    call nn_compute(x, p, nn_tab)

  end subroutine nn_predict

  subroutine relu(x, relu_values)
    !> ReLU activation function
    implicit none
    
    double precision, dimension(:), intent(in)  :: x
    double precision, intent(out)               :: relu_values(size(x))

    relu_values = max(0.0d0, x)

  end subroutine relu


  subroutine sigmoid(x, sigmoid_values)
    !> Sigmoid activation function
    implicit none

    integer, dimension(1)                       :: s ! Shape of the input array x
    integer                                     :: i
    double precision, dimension(:), intent(in)  :: x
    double precision, dimension(:), intent(out) :: sigmoid_values(size(x))

    ! Get shape of array
    s = shape(x)

    ! Fill array with sigmoid values
    do i = 1, s(1)
      sigmoid_values(i) = 1.0d0 / (1.0d0 + dexp(-x(i)))
    end do

  end subroutine sigmoid

  ! Predict refers to the neural network. Given x and the weight matrices of the neural net, the subroutine predicts the pressure
  subroutine nn_compute(x, p, nn_tab_in)
    implicit none

    double precision, intent(in)  :: x(INPUT_SIZE)      ! Input of the NN
    double precision, intent(out) :: p         ! Pressure as return value (scalar) 
    type(nn_table_t), intent(in)  :: nn_tab_in ! Neural network parameters

    double precision              :: xx(HIDDEN_SIZE_1)  ! intermediate result, after first hidden layer computaiton
    double precision              :: yy(HIDDEN_SIZE_1)  ! intermediate result first hidden layer after activation function
    double precision              :: xxx(HIDDEN_SIZE_2) ! intermediate result, after first second layer computaiton
    double precision              :: yyy(HIDDEN_SIZE_2) ! intermediate result second hidden layer after activation function
    double precision              :: y(OUTPUT_SIZE)               ! Output NN as array
    
    ! Do the calculation:
    xx  = matmul(nn_tab_in%weight0, x)   + nn_tab_in%bias0(:,1)
    if (USE_SIGMOID) then
      call sigmoid(xx, yy)
    else 
      call relu(xx, yy)
    end if
    xxx = matmul(nn_tab_in%weight2, yy)  + nn_tab_in%bias2(:,1)
    if (USE_SIGMOID) then
      call sigmoid(xxx, yyy)
    else 
      call relu(xxx, yyy)
    end if
    y   = matmul(nn_tab_in%weight4, yyy) + nn_tab_in%bias4(:,1)

    ! Get the end result as a scalar, not an array
    p = y(1)
  end subroutine nn_compute

  
end module mod_grhd_phys_neuralnet
