!**************************************************
      subroutine cg(maxit,nintf,neq,niut,maxmt1,myrank,ia,ja,ju,si,ri,sintf,rintf,iut,au,arhsu,solu)

      implicit none
      include 'mpif.h'

      integer maxit,maxmt1,nintf,nnode,niut,myrank
      integer i,j,k,i0,nbd,n,nvar,kk,jj,nn
      integer neq,nelem
      integer ju0,jv0,jw0,jp

      integer jp1,jp2,j1,j2,j3
      integer Allocatestatus,iter
      integer alstatus,status(MPI_STATUS_SIZE),tag,ierr

      real v1u(neq),solu(neq)

      real bknum,bkden,akden,bk,ak,ddot,ro,ro0,bknum1,akden1
      real eps,err0,err1,dnrm2,err,dtinv,x0,x1,xcoor,tmp,r1,ro1

      real,dimension(:),allocatable::r,z,p0,svar,rvar

!-----array for CSR matrix
      integer ia(nintf+1),ju(nintf),ja(maxmt1)
      integer si(niut+1),ri(niut+1),iut(niut),request(2*niut)
      integer sintf(si(niut+1)-1),rintf(neq-nintf)
      real au(maxmt1),arhsu(neq)

      n=neq

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
      do i=1,niut
        call MPI_ISEND(svar(si(i)),si(i+1)-si(i), MPI_DOUBLE_PRECISION,iut(i)-1,tag, MPI_COMM_WORLD,request(i),ierr)
        call MPI_IRECV(rvar(ri(i)),ri(i+1)-ri(i), MPI_DOUBLE_PRECISION,iut(i)-1,tag, MPI_COMM_WORLD,request(i+niut),ierr)
      enddo
      do i=1,niut
        call MPI_WAIT(request(i+niut),status,ierr)
      enddo

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
!choi---calculmate coefficient bk and direction vector p
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
!choi    calculmate coefficient ak, new itermate x, new residual r

!-----Exchange p0 to compute A*p0
        do i=1,niut
	  call MPI_WAIT(request(i),status,ierr)
	enddo
	do i=1,si(niut+1)-1
	  svar(i)=p0(sintf(i))
	enddo

        tag=1
	do i=1,niut
	  call MPI_ISEND(svar(si(i)),si(i+1)-si(i), MPI_DOUBLE_PRECISION,iut(i)-1,tag, MPI_COMM_WORLD,request(i),ierr)
      call MPI_IRECV(rvar(ri(i)),ri(i+1)-ri(i), MPI_DOUBLE_PRECISION,iut(i)-1,tag, MPI_COMM_WORLD,request(i+niut),ierr)
    enddo
	do i=1,niut
	  call MPI_WAIT(request(i+niut),status,ierr)
	enddo
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

        err=0.d0
        do i=1,nintf
          err=err+r(i)*r(i)
        enddo
        call MPI_ALLREDUCE(err,err1,1,MPI_DOUBLE_PRECISION, MPI_SUM,MPI_COMM_WORLD,ierr)
        err=dsqrt(err1)
!---------------------------------------------------------
       if (abs(err).gt.1.d20) then
          if (myrank.eq.0) then
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
      do i=1,niut
        call MPI_ISEND(svar(si(i)),si(i+1)-si(i), MPI_DOUBLE_PRECISION,iut(i)-1,tag, MPI_COMM_WORLD,request(i),ierr)
        call MPI_IRECV(rvar(ri(i)),ri(i+1)-ri(i), MPI_DOUBLE_PRECISION,iut(i)-1,tag, MPI_COMM_WORLD,request(i+niut),ierr)
      enddo
      do i=1,niut
        call MPI_WAIT(request(i+niut),status,ierr)
      enddo
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
      
!-----------------------------------------------------------------------
!     Y = A * X
!     input:
!       n     = row dimension of A
!       x     = array of length equal to the column dimension of matrix A
!       a, ja, ia = input matrix in compressed sparse row format.
!     output:
!       y     = real array of length n, containing the product y=Ax
!-----------------------------------------------------------------------
      subroutine amux0(nintf,n,maxmt1,x,y,a,ja,ia)
      integer i, k
      integer nintf,n,maxmt1,ja(maxmt1),ia(nintf+1)
      real*8 a(maxmt1),tmp
      real*8 x(n),y(n)
!
      do i= 1,nintf
        tmp = 0.d0
        do k=ia(i),ia(i+1)-1
          tmp = tmp + a(k)*x(ja(k))
        enddo
        y(i) = tmp
      enddo
      return
      end
