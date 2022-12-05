package com.example.kt

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.WorkerParameters
import com.example.kt.data.repo.FileRepository
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect

class UploadWorker(
    appContext: Context,
    workerParams: WorkerParameters,
    private val fileRepository: FileRepository
) : CoroutineWorker(appContext, workerParams) {
    // Store count for unUploaded files
    var unUploadedFilesCount = 0

    // Error Msg
    var errorMsg: String = ""

    override suspend fun doWork(): Result {
        return try {
            uploadFiles()
            Result.success(
                Data.Builder().putAll(
                    mutableMapOf<String, Any>(
                        "unUploadedFilesCount" to unUploadedFilesCount.toString(),
                        "errorMsg" to errorMsg
                    )
                ).build()
            )
        } catch (e: Error) {
            Log.e("UPLOAD_FAILED", e.toString())
            Result.failure(Data.Builder().putString("errorMsg", e.toString()).build())
        }
    }

    suspend fun uploadFiles() {
        // Get unuploaded files
        val ktFiles = fileRepository.getUnUploadedFileRecords()
        var fileCount = 0
        // try to upload each file
        ktFiles.forEach { ktFile ->
            setProgressAsync(Data.Builder().putInt("progress", fileCount).build())
            fileRepository.uploadFile(ktFile)
                .catch { e ->
                    errorMsg = e.message!!
                    unUploadedFilesCount += 1
                }
                .collect()
            fileCount += 1
        }
    }
}