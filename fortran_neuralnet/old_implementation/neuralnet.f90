module neuralnet
  !> This is the old, original implementation of the neural network within Gmunu
  implicit none

  !> Variables for the neural networks to run:
  type nn_table_t
     integer :: nr_w0, nc_w0
     integer :: nr_b0, nc_b0
     integer :: nr_w2, nc_w2
     integer :: nr_b2, nc_b2
     integer :: nr_w4, nc_w4
     integer :: nr_b4, nc_b4
     double precision, dimension(:,:), allocatable :: weight0
     double precision, dimension(:,:), allocatable :: bias0
     double precision, dimension(:,:), allocatable :: weight2
     double precision, dimension(:,:), allocatable :: bias2
     double precision, dimension(:,:), allocatable :: weight4
     double precision, dimension(:,:), allocatable :: bias4
  end type nn_table_t

  type(nn_table_t) :: nn_tab

  ! Public methods
  public :: grhd_phys_neuralnet_init
  public :: nn_predict

contains

  !> Initialize the module
  subroutine grhd_phys_neuralnet_init(filepath)
    character(*), intent(in)   :: filepath 
    character(len=256)         :: fname = '' 

    !> Initialize the neural network weights and biases here by reading the CSV files

    ! fixme: where are these values come form?
    nn_tab%nr_w0 = 504; nn_tab%nc_w0 =   3
    nn_tab%nr_b0 = 504; nn_tab%nc_b0 =   1
    nn_tab%nr_w2 = 127; nn_tab%nc_w2 = 504
    nn_tab%nr_b2 = 127; nn_tab%nc_b2 =   1
    nn_tab%nr_w4 =   1; nn_tab%nc_w4 = 127
    nn_tab%nr_b4 =   1; nn_tab%nc_b4 =   1

    fname = trim(adjustl(filepath))//"/weight0_flat.csv"
    call read_matrix(fname, nn_tab%nr_w0, nn_tab%nc_w0, nn_tab%weight0)
    fname = trim(adjustl(filepath))//"/bias0_flat.csv"
    call read_matrix(fname, nn_tab%nr_b0, nn_tab%nc_b0, nn_tab%bias0)

    fname = trim(adjustl(filepath))//"/weight2_flat.csv"
    call read_matrix(fname, nn_tab%nr_w2, nn_tab%nc_w2, nn_tab%weight2)
    fname = trim(adjustl(filepath))//"/bias2_flat.csv"
    call read_matrix(fname, nn_tab%nr_b2, nn_tab%nc_b2, nn_tab%bias2)
    
    fname = trim(adjustl(filepath))//"/weight4_flat.csv"
    call read_matrix(fname, nn_tab%nr_w4, nn_tab%nc_w4, nn_tab%weight4)
    fname = trim(adjustl(filepath))//"/bias4_flat.csv"
    call read_matrix(fname, nn_tab%nr_b4, nn_tab%nc_b4, nn_tab%bias4)

  end subroutine grhd_phys_neuralnet_init

  !> Subroutine which reads the values from a CSV and loads them into a matrix
  subroutine read_matrix(fname, nrows, ncols, matrix)
    implicit none

    ! Declare the variables
    character(len=256), intent(in)             :: fname
    integer, intent(in)                        :: nrows, ncols
    double precision, dimension(:, :), allocatable, intent(out) :: matrix

    ! local vars
    integer                                    :: iflag, nlines
    integer                                    :: i, j
    double precision, dimension(nrows*ncols)   :: values
    ! To reshape in the correct shape after reading:
    integer, dimension (1:2) :: order2 = (/ 2, 1 /)

    ! Allocate memory for matrix
    allocate(matrix(nrows, ncols))

    ! Open the file for reading
    open(unit=666, file=fname, status='old')!, access='sequential')  !
    do i = 1, nrows*ncols
      read(666,*,iostat=iflag) values(i)
      if (iflag/=0) exit
    end do
    ! Reading is over, close the file
    close(unit = 666)

    matrix = reshape(values, (/ nrows, ncols /), order=order2)
    !matrix = transpose(matrix)

  end subroutine read_matrix

  !> Neural network calculations:
  !> TODO -put these in a separate module!

  !> Given the conserved variables D, S, tau, returns the pressure p as computed by the neural network
  subroutine nn_predict(D, S, tau, p)
    implicit none

    !> Input for this subroutine: the conserved variables
    double precision, intent(in) :: D   !< Conserved energy density
    double precision, intent(in) :: S   !< Conserved momentum density
    double precision, intent(in) :: tau !< Conserved energy density relative to D
    !> Declare auxiliary matrices (weights and biases). We are using two hidden layers, with sizes specified below:
    
    !> Output for this subroutine: the pressure, computed by the neural networke
    double precision, intent(out) :: p

    ! Input x for the neural net, is written as a vector here: (D, S, tau)
    double precision, dimension(3) :: x

    ! Declare other variables, such as filename to read the CSV files and integers i, j for loops
    !character(len=256) :: fname
    integer            :: i, j
 
    x(1) = D
    x(2) = S
    x(3) = tau
    ! Do the neural network computation, returns the pressure
    call nn_compute(x, p, nn_tab)

  end subroutine nn_predict

  !> This is a subroutine which takes input values x as an array and computes their sigmoid function values for the neural network
  subroutine sigmoid(x, sigmoid_values)
    implicit none

    ! Shape of the array
    integer, dimension(1) :: s
    integer :: i
    double precision, dimension(:), intent(in)  :: x
    double precision, dimension(:), intent(out) :: sigmoid_values

    s = shape(x)

    ! Fill array with sigmoid values
    do i = 1, s(1)
      sigmoid_values(i) = 1.0d0 / (1.0d0 + dexp(-x(i)))
    end do

  end subroutine sigmoid

  ! Predict refers to the neural network. Given x and the weight matrices of the neural net, the subroutine predicts the pressure
  subroutine nn_compute(x, p, nn_tab_in)
    implicit none

    double precision, intent(in)  :: x(3)
    double precision, intent(out) :: p           ! pressure, but as scalar value
    type(nn_table_t), intent(in)  :: nn_tab_in

    double precision              :: xx(504)  ! intermediate result 1, after first hidden layer
    double precision              :: yy(504)  ! intermediate result 1, applied sigmoid
    double precision              :: xxx(127) ! intermediate result 2, after second hidden layer
    double precision              :: yyy(127) ! intermediate result 2, after sigmoid
    double precision              :: y(1)     ! pressure, but as array
    
    ! Do the calculation:
    xx  = matmul(nn_tab_in%weight0, x)   + nn_tab_in%bias0(:,1)
    call sigmoid(xx, yy)
    xxx = matmul(nn_tab_in%weight2, yy)  + nn_tab_in%bias2(:,1)
    call sigmoid(xxx, yyy)
    y   = matmul(nn_tab_in%weight4, yyy) + nn_tab_in%bias4(:,1)

    ! Get the end result as a scalar, not an array
    p = y(1)
  end subroutine nn_compute

  
end module neuralnet
