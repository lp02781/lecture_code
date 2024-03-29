PROGRAM mpi_cg
	include 'mpif.h'

	integer myrank, size_Of_Cluster, ierror, tag, maxit
	integer p, n, cnt, nnz, m, nintf, neq, niut, maxmt1
	real n_dum, h, s, rhs, m_dum, power
	integer, dimension(:), allocatable :: ia,ja,ju,si,ri,sintf,rintf,iut
	real, dimension(:), allocatable :: u,b,r,t,au,arhsu,solu
	
	call MPI_INIT(ierror)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, size_Of_Cluster, ierror)
	call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierror)
	
	
	!print *, 'Hello World from process: ', p, 'of ', size_Of_Cluster
	p = 3
	n = 9
	power = 2
	
	!nnz = (n-2)*3+3
	
	n_dum = n
	
	m_dum = n/p
	m = m_dum
	
	nnz = m*3
		
	h = 1/n_dum
	s = 7
	rhs = -s*h**power
	
	if (myrank.eq.0) then
		nnz = nnz-1
	else if(myrank.eq.p-1) then
		nnz=nnz-2
	else
		nnz = nnz
	end if
		
	allocate(au(1:nnz), ia(1:m+1), ja(1:nnz), ju(1:m)) 
	allocate(b(1:m),u(1:m),r(1:n),t(1:n)) 
	
	do i=1, nnz
		au(i) = 0
		ja(i) = 0
	enddo
	
	do i=1, m
		b(i) = 0
		u(i) = 0
		ju(i) = 0
	enddo

	do i=1, n
		r(i) = 0
		t(i) = 0
	enddo
	
	do i=1, m+1
		ia(i) = 0
	enddo
	
	if (myrank.eq.0) then
		b(1) = 0
		u(1) = 0
		au = 0
		au(1) = 1
		ia(1) = 1
		ja(1) = 1
		ju(1) = 1
		au(2) = -1
		ja(2) = 2
		cnt = 3
		do i = 2, m
			au(cnt) = -1
			au(cnt+1) = 2
			au(cnt+2) = -1
			ju(i) = cnt +1
			ia(i) = cnt
			u(i) = 0
			b(i) = rhs
			ja(cnt) = i-1
			ja(cnt+1) = i
			ja(cnt+2) = i+1
			cnt = cnt + 3
		end do
		if(p==1) then
			cnt = cnt -3
			au(cnt)=1
			ia(m) = cnt
			ja(cnt) = m
			ju(m) = cnt
			ia(m+1) = cnt+1
			u(m) = 1
			b(m) = 1
			iut = 1
			do i =1,m
				t(i) = u(i)
			end do
			go to 10
		end if	
		ia(m+1) = cnt
		cnt = cnt -1
		niut = 1	
	else if(myrank.eq.p-1) then
		cnt = 1
		do i=1, m-1
			au(cnt) = -1
			au(cnt+1) = 2
			au(cnt+2) = -1
			ju(i) = cnt+1
			ia(i) = cnt
			u(i) = 0
			b(i) = rhs
			ja(cnt) = i-1
			ja(cnt+1) = i
			ja(cnt+2) = i+1
			cnt = cnt+3
		end do
		au(cnt) = 1
		ia(m) = cnt
		ja(cnt) = m
		ju(m) = cnt
		ia(m+1) = cnt+1
		u(m) = 1
		b(m) = 1
		niut = 1
	else
		cnt = 1
		do i=1, m
			au(cnt) = -1
			au(cnt+1) = 2
			au(cnt+2) = -1
			ju(i) = cnt+1
			ia(i) = cnt
			u(i) = 0
			b(i) = rhs
			ja(cnt) = i-1
			ja(cnt+1) = i
			ja(cnt+2) = i+1
			cnt = cnt +3
		end do
		ia(m+1) = cnt
		niut = 2
	end if
	10 continue
	
	nintf = nnz - niut
	maxit = 1000000
	maxmt1 = nnz
	neq = n
	allocate(si(0:(niut+1)), ri(0:niut+1)) 
	allocate(sintf(0:niut+1-1), rintf(0:(neq-nintf)), iut(0:niut))
	!allocate(sintf(0:(si(niut+1)-1)), rintf(0:(neq-nintf)), iut(0:niut))
	!allocate(arhsu(0:neq), solu(0:neq))
	
	do i=0, niut+1
		si(i)=0
		ri(i)=0
	end do
	
	do i=0, niut
		sintf(i)=0
		rintf(i)=0
		iut(i)=0
	end do
	print *, myrank, 'haha'
	
	!call cg(maxit,nintf,neq,niut,maxmt1,myrank,ia,ja,ju, si,ri,sintf,rintf,iut,au,b,u)
	!call cg(maxit,nintf,neq,niut,maxmt1,myrank,ia,ja,ju, si,ri,sintf,rintf,iut,au,arhsu,solu)
	call MPI_FINALIZE(ierror)
END

!**************************************************
      subroutine cg(maxit,nintf,neq,niut,maxmt1,myrank,ia,ja,ju, si,ri,sintf,rintf,iut,au,arhsu,solu)


      !implicit none
      include 'mpif.h'

      integer maxit,maxmt1,nintf,nnode,niut,myrank
      integer i,j,k,i0,nbd,n,nvar,kk,jj,nn
      integer neq,nelem
      integer ju0,jv0,jw0,jp

      integer jp1,jp2,j1,j2,j3
      integer Allocatestatus,iter
      integer alstatus,status(MPI_STATUS_SIZE),tag,ierr

      real v1u(neq),solu(0:neq)

      real bknum,bkden,akden,bk,ak,ddot,ro,ro0,bknum1,akden1
      real eps,err0,err1,dnrm2,err,dtinv,x0,x1,xcoor,tmp,r1,ro1

      real,dimension(:),allocatable::r,z,p0,svar,rvar

!-----array for CSR matrix
      integer ia(nintf+1),ju(nintf),ja(maxmt1)
      integer si(niut+1),ri(niut+1),iut(niut),request(2*niut)
      integer sintf(si(niut+1)-1),rintf(neq-nintf)
      real au(maxmt1),arhsu(neq)

      n=neq
      nnod = neq

      Allocate(r(n),z(n),p0(n),stat=Allocatestatus)
      if(AllocateStatus/=0) stop "**Not enough memory "
      
!----Predictor  By Diag. pc
      r=0.d0 !! in fact, this is not necessary.
      do i=1,nintf
        solu(i)=arhsu(i)/au(ju(i))
      enddo

!-----Exchange solu to compute A*solu
      allocate(svar(si(niut+1)-1),stat=alstatus)
      if (alstatus/=0) stop 'not enough 12memory'
      allocate(rvar(ri(niut+1)-1),stat=alstatus)
      if (alstatus/=0) stop 'not enough 13memory'
      
      do i=1,si(niut+1)-1
        svar(i)=solu(sintf(i))
      enddo
      tag=1
      
      call mpi_barrier(mpi_comm_world,ierr)
	  if(myrank.eq.0 .and. p.gt.1) then
			call MPI_ISEND(solu(nintf),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_IRECV(solu(nintf+1),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(2),ierr)
	  else if(myrank.eq.p-1 .and. p.gt.1) then
			call MPI_ISEND(solu(1),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_IRECV(solu(0),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(2),ierr)
      else if(p.gt.1) then
			call MPI_ISEND(solu(1),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_ISEND(solu(nintf),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(2),ierr)
			call MPI_IRECV(solu(0),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(3),ierr)
			call MPI_IRECV(solu(nintf+1),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(4),ierr)
	  end if
      if(p.gt.1) then
		do i=1,niut
			call MPI_WAIT(request(i+niut),status,ierr)
		enddo
	  endif
	  call mpi_barrier(mpi_comm_world,ierr)

      do i=1,n-nintf
        solu(rintf(i))=rvar(i)
      enddo

      call amux0(nintf,n,maxmt1,solu,r,au,ja,ia)
!---------------------------------------------------
      do i=1,nintf
       r(i)=arhsu(i)-r(i)
      enddo
!-----Diag PC z=r/Diag(*)
      do i=1,nintf
       z(i)=r(i)/au(ju(i))
      enddo
		
      err0 = 0.d0
      do i=1,nintf
        err0=err0+r(i)*r(i)
      enddo

      call MPI_ALLREDUCE(err0,err1,1,MPI_DOUBLE_PRECISION, MPI_SUM,MPI_COMM_WORLD,ierr)
      err0=sqrt(err1)
      if ( err0.eq.0.0 ) goto 502

      do iter=1,maxit
        tag = iter 
!-----calculmate coefficient bk and direction vector p
        bknum=0.d0
        do i=1,nintf
          bknum=bknum+r(i)*z(i)
        enddo
        call MPI_ALLREDUCE(bknum,bknum1,1,MPI_DOUBLE_PRECISION, MPI_SUM,MPI_COMM_WORLD,ierr)
        bknum = bknum1

        if ( iter.eq.1) then
         p0=z
        else
         bk = bknum/bkden
         p0=z+bk*p0
        endif
        bkden = bknum
!    calculmate coefficient ak, new itermate x, new residual r

!-----Exchange p0 to compute A*p0
        do i=1,niut
	  call MPI_WAIT(request(i),status,ierr)
	enddo
	do i=1,si(niut+1)-1
	  svar(i)=p0(sintf(i))
	enddo

        tag=1
	  
	  call mpi_barrier(mpi_comm_world,ierr)
	  if(myrank.eq.0 .and. p.gt.1) then
			call MPI_ISEND(solu(nintf),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_IRECV(solu(nintf+1),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(2),ierr)
	  else if(myrank.eq.p-1 .and. p.gt.1) then
			call MPI_ISEND(solu(1),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_IRECV(solu(0),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(2),ierr)
      else if(p.gt.1) then
			call MPI_ISEND(solu(1),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_ISEND(solu(nintf),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(2),ierr)
			call MPI_IRECV(solu(0),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(3),ierr)
			call MPI_IRECV(solu(nintf+1),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(4),ierr)
	  end if
      if(p.gt.1) then
		do i=1,niut
			call MPI_WAIT(request(i+niut),status,ierr)
		enddo
	  endif
	  call mpi_barrier(mpi_comm_world,ierr)
	  
	do i=1,ri(niut+1)-1
	  p0(rintf(i))=rvar(i)
	enddo

        call amux0 (nintf,n,maxmt1,p0,z,au,ja,ia) !!A*p0=z
!-----------------------------------------------------
        akden=0.d0
        do i=1,nintf
         akden=akden+p0(i)*z(i)
        enddo
        call MPI_ALLREDUCE(akden,akden1,1,MPI_DOUBLE_PRECISION, MPI_SUM,MPI_COMM_WORLD,ierr)
        ak = bknum/akden1

        do i=1,nintf
          solu(i) = solu(i) + ak*p0(i)
          r(i) = r(i) - ak*z(i)
        enddo
!-------Diag PC z=r/Diag(*)
        do i=1,nintf
         z(i)=r(i)/au(ju(i))
        enddo
		
		!error=0 
		!error0=0
		!do i, n
			!error0=t0(i)-t(i)
			!error=error+error0*error0
		!end do
		!error = sqrt(error)
	  
        err=0.d0
        do i=1,nintf
          err=err+r(i)*r(i)
        enddo
        call MPI_ALLREDUCE(err,err1,1,MPI_DOUBLE_PRECISION, MPI_SUM,MPI_COMM_WORLD,ierr)
        err=sqrt(err1)
!---------------------------------------------------------
       if (abs(err).gt.1.d20) then
          if (myrank.eq.0) then
!			 fname = 'T'
!			 fp1 = char(nnode/100000+48)
!			 fp2 = char(mod(nnode,100000)/10000+48)
!			 fp3 = char(mod(nnode,10000)/1000+48)
!			 fp4 = char(mod(nnode,1000)/100+48)
!			 fp5 = char(mod(nnode,100)/10+48)
!			 fp6 = char(mod(nnode,10)/48)
!             Ext = .dat
!             fout = fname//fp1//fp2//fp3//fp4//fp5//fp6//Ext
!             open(15,file=fout, status='unknown')
!             write(15,*)'varibles="x","Temperature"'
!             do i=1,m
!				t(i) = u(i)
!			 end do
!			 tag = 1
!			 do ip=2,p
!				do i=1,m
!					call MPI_RECV(u0(i),1,MPI_DOUBLE_PRECISION,ip-1,tag,MPI_COMM_WORLD,status,ierr)
!					j=(ip-1)*m+i
!					t(j)=u0(i)
!				end do
!			 end do
!			 do i =1,nnode
!				write(15,*) coord(:,i),t(i)
!			 end do
!			 close(15)
!		 else 
!			tag = 1
!			do i = 1,m
!				call MPI_SEND(u(i),1,MPI_DOUBLE_PRECISION,0,tag,MPI_COMM_WORLD,ierr)
!			end do 
             
             write(*,*) 'blow-up in the solver, res=',err
         endif  
         stop
       endif
       if(myrank.eq.0)write(*,*)'Iter=',iter,'Residual=',err/err0
       if ( err/err0.le.1.d-6 .and. iter.ge.10 ) go to 502
!---------------------------------------------------------
      enddo ! Main -loop
502   continue

      do i=1,niut
        call MPI_WAIT(request(i),status,ierr)
      enddo
      do i=1,si(niut+1)-1
        svar(i)=solu(sintf(i))
      enddo
      tag=1
      call mpi_barrier(mpi_comm_world,ierr)
	  if(myrank.eq.0 .and. p.gt.1) then
			call MPI_ISEND(solu(nintf),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_IRECV(solu(nintf+1),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(2),ierr)
	  else if(myrank.eq.p-1 .and. p.gt.1) then
			call MPI_ISEND(solu(1),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_IRECV(solu(0),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(2),ierr)
      else if(p.gt.1) then
			call MPI_ISEND(solu(1),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(1),ierr)
			call MPI_ISEND(solu(nintf),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(2),ierr)
			call MPI_IRECV(solu(0),1,MPI_DOUBLE_PRECISION,myrank-1,tag,MPI_COMM_WORLD,request(3),ierr)
			call MPI_IRECV(solu(nintf+1),1,MPI_DOUBLE_PRECISION,myrank+1,tag,MPI_COMM_WORLD,request(4),ierr)
	  end if
      if(p.gt.1) then
		do i=1,niut
			call MPI_WAIT(request(i+niut),status,ierr)
		enddo
	  endif
	  call mpi_barrier(mpi_comm_world,ierr)
      do i=1,n-nintf
        solu(rintf(i))=rvar(i)
      enddo

      if(myrank.eq.0)write(*,*)'Iter=',iter,'Residual=',err/err0
      do i=1,niut
        call MPI_WAIT(request(i),status,ierr)
      enddo

      Deallocate(r,z,p0,svar,rvar)

      return
      end

!***********************************************************************
      subroutine amux0(nintf,n,maxmt1,x,y,a,ja,ia)
!-----------------------------------------------------------------------
!     Y = A * X
!     input:
!       n     = row dimension of A
!       x     = array of length equal to the column dimension of matrix A
!       a, ja, ia = input matrix in compressed sparse row format.
!     output:
!       y     = real array of length n, containing the product y=Ax
!-----------------------------------------------------------------------
      integer i, k
      integer nintf,n,maxmt1,ja(maxmt1),ia(nintf+1)
      real a(maxmt1),tmp
      real x(0:n-1),y(0:n-1)

      do i= 1,nintf
        tmp = 0.d0
        do k=ia(i),ia(i+1)-1
          tmp = tmp + a(k)*x(ja(k))
        enddo
        y(i) = tmp
      enddo
      return
      end
