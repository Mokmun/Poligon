import React from 'react'
import Link from 'next/link'
import Image from 'next/image'

const main =  () => {
  return (
    <div className="relative flex flex-col-reverse md:flex-row w-full items-center md:items-start px-16 md:pb-24 justify-center h-screen md:h-dvh">
        <div className="flex flex-col items-center md:items-start justify-center h-full w-full md:w-3/5">
          <div className="flex flex-col items-center md:items-start gap-6 justify-start w-full">
            <h1 className="text-3xl md:text-5xl font-bold text-center md:text-left">Build your 3D model</h1>
            <p className="text-white text-sm md:text-base/7 tracking-wide text-center md:text-left md:px-0">
              Create and view 3D models
              and explore a wide range of furniture designs. Our platform allows you to customize and visualize your own creations in a 3D environment, 
              making it easier to bring your ideas to life.
            </p>
          </div>
          <div className='flex flex-row gap-4 mt-6 md:mt-10'>
            <button className="btn-primary text-white text-sm md:text-lg px-6 md:px-9 py-2 md:py-3 rounded-xl">
              <Link href='/create3D'>
                Start Generating
              </Link>
            </button>
          </div>
        </div>
        <Image src='/assets/star.png' alt="hero" width={400} height={400} className='object-contain mt-6 md:mt-0 md:pt-24' />
    </div>
  )
}

export default main