package com.example.kt.data.repo

import android.net.Uri
import androidx.core.net.toFile
import com.example.kt.data.local.daos.FileDao
import com.example.kt.data.models.KTFile
import com.example.kt.service.FileAPI
import kotlinx.coroutines.flow.flow
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import javax.inject.Inject

class FileRepository @Inject constructor(
    private val fileDao: FileDao,
    private val fileAPI: FileAPI
){
    // insert new file record
    suspend fun insertNewFileRecord(uri: String) { fileDao.insertFileRecord(uri) }

    // get un-uploaded file records
    suspend fun getUnUploadedFileRecords(): List<KTFile> { return fileDao.getUnUploadedFiles() }

    // upload file
    suspend fun uploadFile(uri: String) = flow {
        val fileUri = Uri.parse(uri)
        // Get file
        val file = fileUri.toFile()
        val fileName = file.name
        val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
        val filePart = MultipartBody.Part.createFormData("file", fileName, requestFile)
        val response = fileAPI.uploadFile(filePart)
        val responseBody = response.body()

        if (!response.isSuccessful) {
            error("Failed to upload file")
        }

        if (responseBody != null) {
            emit(responseBody)
        } else {
            error("Request failed, response body was null")
        }
    }
}