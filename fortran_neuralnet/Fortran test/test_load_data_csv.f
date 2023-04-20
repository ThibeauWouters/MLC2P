program read_data
  !use csv_module ! can we use/import this?
  implicit none

  ! Declarations
  real, dimension(:,:), allocatable :: A
  character(len=:), allocatable     :: buffer
  integer                           :: file_unit, size_of_file, rc
  integer                           :: i
  character(len=256)                :: filename = "NNC2Pv0_params_stack.0.weight.csv"
  logical                           :: file_exists

  ! Open file.
  !access  = 'stream', & ! other option

  open (action  = 'read', &
        file    = filename, &
        form    = 'formatted', &
        iostat  = rc, &
        newunit = file_unit, &
        status = 'old')
  if (rc /= 0) stop 'Error: opening file failed'

  ! Get file size and allocate buffer string.
  !inquire (file_unit, size=size_of_file)
  !allocate (character(len=size_of_file) :: buffer)
  !read (file_unit, iostat=rc) buffer

  ! Read file into buffer.
  do i=1,2
    read (file_unit, *, iostat=rc) A(i,:) ! read a line
    print *, rc
  end do
  close (file_unit)
  print *, "Read the file!"
  print *, i
  print *, A



  ! test: this prints the first 100 values of the COLUMN, so no rows
  !do i=1,5
  !  read(10,*) line
  !  print *, line
  !end do

  !close(10)

  ! Print the matrix
  !print *, A
end program read_data
