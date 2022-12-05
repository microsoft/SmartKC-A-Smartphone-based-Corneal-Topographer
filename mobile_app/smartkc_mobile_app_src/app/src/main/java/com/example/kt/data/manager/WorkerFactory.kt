package com.example.kt.data.manager

import android.content.Context
import androidx.work.ListenableWorker
import androidx.work.WorkerFactory
import androidx.work.WorkerParameters
import com.example.kt.UploadWorker
import com.example.kt.data.repo.FileRepository

class WorkerFactory(private val fileRepository: FileRepository): WorkerFactory() {
    override fun createWorker(
        appContext: Context,
        workerClassName: String,
        workerParameters: WorkerParameters
    ): ListenableWorker? {

        return when (workerClassName) {
            UploadWorker::class.java.name ->
                UploadWorker(
                    appContext,
                    workerParameters,
                    fileRepository
                )
            else ->
                // Return null, so that the base class can delegate to the default WorkerFactory.
                null
        }
    }
}