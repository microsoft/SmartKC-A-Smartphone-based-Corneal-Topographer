package com.example.kt

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.example.kt.data.repo.FileRepository

class UploadWorker(
    appContext: Context,
    workerParams: WorkerParameters,
    private val fileRepository: FileRepository
): CoroutineWorker(appContext, workerParams) {
    override suspend fun doWork(): Result {
        TODO("Not yet implemented")
    }
}